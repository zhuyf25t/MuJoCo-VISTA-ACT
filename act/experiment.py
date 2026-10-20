"""Immutable epoch evaluation and best.pt selection by closed-loop task results."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

import torch

from .checkpoint import load_checkpoint, save_checkpoint, deployment_metadata
from .common import atomic_json, digest, file_digest, read_json
from .task_eval import (evaluate_checkpoint, task_score, SCORE_NAMES, SCORE_VERSION,
                        VALIDATION_SEEDS, TASK_EXECUTION, FRONT_RENDER_PROTOCOL)


def _source_files():
    repo = Path(__file__).resolve().parents[1]
    files = set(repo.glob('*.py'))
    for directory in ('act', 'sim', 'control'):
        files.update((repo / directory).rglob('*.py'))
    files.update((repo / 'models').rglob('*.xml'))
    files.update((repo / 'act' / 'configs').glob('*.yaml'))
    files.add(repo / 'requirements.txt')
    return repo, [path for path in sorted(files)
                  if path.is_file() and path.resolve().is_relative_to(repo)
                  and '__pycache__' not in path.parts
                  and path.name not in ('config_local.py', 'config.local.py')]


def archive_source(experiment):
    """Archive source/XML for provenance; source changes do not block resume."""
    repo, files = _source_files()
    destination = Path(experiment) / 'source_runs'
    destination.mkdir(parents=True, exist_ok=True)
    number = 1
    while (destination / f'run_{number:03d}.zip').exists():
        number += 1
    archive = destination / f'run_{number:03d}.zip'
    hashes = {}
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            relative = path.relative_to(repo).as_posix()
            content = path.read_bytes()
            hashes[relative] = hashlib.sha256(content).hexdigest()
            bundle.writestr(relative, content)
    manifest = archive.with_suffix('.json')
    atomic_json(manifest, {'archive': str(archive.resolve()), 'archive_sha256': file_digest(archive),
                           'source_sha256': digest(hashes), 'files': hashes})
    return str(manifest.resolve())


class EpochEvaluator:
    def __init__(self, seeds=VALIDATION_SEEDS, device='auto', video_every=0, workers=10):
        self.seeds, self.device = list(seeds), device
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
                or any(type(seed) is not int or seed < 0 for seed in self.seeds)):
            raise ValueError('Supply distinct nonnegative evaluation seeds')
        if type(workers) is not int or workers < 1:
            raise ValueError('Evaluation workers must be a positive integer')
        if type(video_every) is not int or video_every < 0:
            raise ValueError('video_every must be a nonnegative integer')
        self.workers, self.video_every = workers, video_every
        self.protocol = {'version': 2, 'seeds': self.seeds, 'max_steps': 600,
                         'chunk_size': 32, 'workers': workers,
                         'delivery_stable_seconds': .5,
                         'execution_config': dict(TASK_EXECUTION),
                         'render_protocol': FRONT_RENDER_PROTOCOL, 'render_warmup_frames': 1,
                         'score_names': list(SCORE_NAMES), 'score_version': SCORE_VERSION}
        self.protocol_sha256 = digest(self.protocol)
        self.source_manifest = None

    def freeze_protocol(self, experiment):
        experiment = Path(experiment)
        experiment.mkdir(parents=True, exist_ok=True)
        path = experiment / 'task_eval_protocol.json'
        if path.exists():
            saved = read_json(path)
            if any(saved.get(key) != value for key, value in self.protocol.items()):
                raise ValueError('Task evaluation protocol differs; start a new experiment')
            # Existing K32/H8 runs may include source and display fields. Keep
            # their document identity so already committed reports stay usable.
            self.protocol_sha256 = digest(saved)
        else:
            artifacts = ('task_metrics.jsonl', 'best.pt', 'best.json', 'task_best.pt', 'task_best.json')
            if (any((experiment / name).exists() for name in artifacts)
                    or any(experiment.glob('eval_epoch_*'))):
                raise ValueError('Existing evaluation has no matching v2 protocol; start a new experiment')
            atomic_json(path, self.protocol)
            self.protocol_sha256 = digest(self.protocol)

    @staticmethod
    def _records(experiment):
        path = experiment / 'task_metrics.jsonl'
        if not path.exists():
            return []
        lines = path.read_text(encoding='utf-8').splitlines()
        records = []
        for index, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if index != len(lines) - 1:
                    raise ValueError('Corrupt task metric history; start a new experiment') from exc
                # A crash may leave only the final append incomplete.
        return records

    @staticmethod
    def _write_records(experiment, records):
        log = experiment / 'task_metrics.jsonl'
        temporary = log.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(''.join(json.dumps(row, allow_nan=False) + '\n' for row in records))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, log)

    @staticmethod
    def _require_checkpoint(checkpoint):
        if ('deployment' not in checkpoint or deployment_metadata(checkpoint) != TASK_EXECUTION
                or type(checkpoint.get('config', {}).get('model', {}).get('chunk_size')) is not int
                or checkpoint['config']['model']['chunk_size'] != 32):
            raise ValueError('Task selection requires saved K32/H8 deployment with ensemble off')
        if type(checkpoint.get('epoch')) is not int or checkpoint['epoch'] < 0:
            raise ValueError('Task selection requires a nonnegative checkpoint epoch')

    @classmethod
    def _snapshot(cls, path, checkpoint):
        cls._require_checkpoint(checkpoint)
        frozen = {key: value for key, value in checkpoint.items() if key not in ('optimizer', 'scaler', 'rng')}
        if not path.exists():
            save_checkpoint(path, frozen)
            return
        saved = load_checkpoint(path)
        cls._require_checkpoint(saved)
        identity = ('epoch', 'global_step', 'config', 'schema', 'stats', 'prepared',
                    'action_codec', 'deployment', 'selection_metric')
        if (any(saved.get(key) != frozen.get(key) for key in identity)
                or saved['model'].keys() != frozen['model'].keys()
                or any(saved['model'][key].dtype != value.dtype
                       or not torch.equal(saved['model'][key].cpu(), value.detach().cpu())
                       for key, value in frozen['model'].items())):
            raise ValueError(f'Immutable epoch snapshot differs from checkpoint: {path}')

    def _score_valid(self, score):
        return (isinstance(score, list) and len(score) == len(SCORE_NAMES)
                and all(type(value) is int and 0 <= value <= len(self.seeds) for value in score))

    def _validate_summary(self, summary, sha, epoch):
        episodes = summary.get('episodes', [])
        if not isinstance(episodes, list) or any(not isinstance(row, dict) for row in episodes):
            raise ValueError('Invalid task episode records; start a new experiment')
        expected = {'checkpoint_sha256': sha, 'checkpoint_epoch': epoch - 1,
                    'checkpoint_epoch_number': epoch, 'execution_config': TASK_EXECUTION,
                    'render_protocol': FRONT_RENDER_PROTOCOL, 'render_warmup_frames': 1}
        invalid_episode = any(
            any(episode.get(key) != value for key, value in expected.items())
            or episode.get('stop_reason') == 'window_closed'
            or type(episode.get('success')) is not bool
            or type(episode.get('final_success')) is not bool
            or episode['success'] != episode['final_success']
            or type(episode.get('deliveries')) is not int or episode['deliveries'] < 0
            or type(episode.get('ever_confirmed_grasp')) is not bool
            for episode in episodes)
        if (summary.get('complete') is not True or summary.get('errors') != []
                or summary.get('missing_seeds') != [] or summary.get('interrupted_seeds') != []
                or any(summary.get(key) != value for key, value in expected.items())
                or summary.get('requested_seeds') != self.seeds or summary.get('completed_seeds') != self.seeds
                or summary.get('requested_episode_count') != len(self.seeds)
                or summary.get('completed_episode_count') != len(self.seeds)
                or summary.get('max_steps') != 600
                or summary.get('workers') != self.workers
                or summary.get('active_workers') != min(self.workers, len(self.seeds))
                or summary.get('process_start_method') != ('spawn' if self.workers > 1 else None)
                or summary.get('task_truth_is_policy_input') is not False
                or summary.get('score_version') != SCORE_VERSION or summary.get('score_names') != list(SCORE_NAMES)
                or summary.get('delivery_stable_seconds') != .5
                or summary.get('task_eval_protocol_sha256') != self.protocol_sha256
                or [episode.get('seed') for episode in episodes] != self.seeds or invalid_episode
                or not self._score_valid(summary.get('score'))
                or list(task_score(episodes)) != summary['score']):
            raise ValueError('Incomplete or incompatible v2 task evaluation cannot select best.pt; start a new experiment for protocol changes')

    def _summary(self, experiment, path, epoch):
        sha = file_digest(path)
        base = experiment / f'eval_epoch_{epoch:03d}'
        for output in (base, *sorted(experiment.glob(f'{base.name}_attempt_*'))):
            report = output / 'task_eval.json'
            if not report.exists():
                continue
            summary = read_json(report)
            if summary.get('checkpoint_sha256') != sha:
                raise ValueError(f'Evaluation does not belong to immutable snapshot: {report}')
            if summary.get('complete') is True:
                if 'task_eval_protocol_sha256' not in summary:
                    # The child evaluator committed before this wrapper could
                    # bind the result to its evaluation protocol. Preserve and rerun.
                    continue
                self._validate_summary(summary, sha, epoch)
                return summary, output
        output, attempt = base, 1
        while output.exists():
            output = experiment / f'{base.name}_attempt_{attempt:03d}'
            attempt += 1
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if self.device == 'auto' else self.device
        print(f'TASK EVAL epoch={epoch} seeds={self.seeds} K32/H8 workers={self.workers}', flush=True)
        summary = evaluate_checkpoint(path, output, seeds=self.seeds, device=device, max_steps=600,
                                      video=self.video_every > 0 and epoch % self.video_every == 0,
                                      workers=self.workers)
        self.freeze_protocol(experiment)
        source = read_json(self.source_manifest) if self.source_manifest is not None else {}
        summary = {**summary, 'evaluation_source_sha256': source.get('source_sha256'),
                   'task_eval_protocol_sha256': self.protocol_sha256,
                   'evaluation_source_manifest': self.source_manifest}
        self._validate_summary(summary, sha, epoch)
        atomic_json(output / 'task_eval.json', summary)
        return summary, output

    def _validate_record(self, record):
        expected = {'seeds': self.seeds, 'workers': self.workers,
                    'execution_config': TASK_EXECUTION, 'render_protocol': FRONT_RENDER_PROTOCOL,
                    'score_names': list(SCORE_NAMES), 'score_version': SCORE_VERSION,
                    'task_eval_protocol_sha256': self.protocol_sha256}
        if (any(record.get(key) != value for key, value in expected.items())
                or not self._score_valid(record.get('score'))
                or type(record.get('epoch')) is not int or record['epoch'] < 1):
            raise ValueError('Cannot compare task scores from different protocols; start a new experiment')

    def _publish(self, experiment, record):
        records = self._records(experiment)
        for existing in records:
            self._validate_record(existing)
            identity = set(existing) | set(record)
            identity -= {'task_eval', 'source_manifest', 'source_sha256'}
            if (existing['epoch'] == record['epoch']
                    and any(existing.get(key) != record.get(key) for key in identity)):
                raise ValueError('Epoch task record differs from its immutable evaluation')
        prior_best = max((tuple(row['score']) for row in records), default=None)
        new_epoch = not any(row['epoch'] == record['epoch'] for row in records)
        improved = new_epoch and (prior_best is None or tuple(record['score']) > prior_best)
        records = [row for row in records if row['epoch'] != record['epoch']] + [record]
        records.sort(key=lambda row: row['epoch'])
        self._write_records(experiment, records)
        best = max(records, key=lambda row: tuple(row['score']))  # Sorted ties retain earliest epoch.
        selected, model_path = experiment / 'best.json', experiment / 'best.pt'
        previous = read_json(selected) if selected.exists() else {}
        if (any(previous.get(key) != value for key, value in best.items()) or not model_path.exists()
                or previous.get('best_sha256') != file_digest(model_path)):
            source = Path(best['checkpoint'])
            if file_digest(source) != best['checkpoint_sha256']:
                raise ValueError('Selected immutable snapshot has changed')
            saved = load_checkpoint(source)
            self._require_checkpoint(saved)
            save_checkpoint(model_path, {**saved, 'task_selection': best})
            atomic_json(selected, {**best, 'best_epoch': best['epoch'], 'best_score': best['score'],
                                   'best_sha256': file_digest(model_path)})
            print(f'Published best.pt epoch={best["epoch"]} score={best["score"]}', flush=True)
        return {**record, 'best_epoch': best['epoch'], 'best_score': best['score'], 'improved': improved}

    def __call__(self, experiment, checkpoint):
        self._require_checkpoint(checkpoint)
        experiment = Path(experiment).resolve()
        self.freeze_protocol(experiment)
        epoch = checkpoint['epoch'] + 1
        archive = experiment / 'checkpoints'
        archive.mkdir(exist_ok=True)
        path = archive / f'epoch_{epoch:03d}.pt'
        self._snapshot(path, checkpoint)
        summary, output = self._summary(experiment, path, epoch)
        record = {'epoch': epoch, 'checkpoint': str(path), 'score': summary['score'],
                  'score_names': list(SCORE_NAMES), 'score_version': SCORE_VERSION,
                  'seeds': self.seeds, 'workers': self.workers,
                  'checkpoint_sha256': summary['checkpoint_sha256'],
                  'execution_config': dict(TASK_EXECUTION), 'render_protocol': FRONT_RENDER_PROTOCOL,
                  'source_sha256': summary.get('evaluation_source_sha256'),
                  'task_eval_protocol_sha256': self.protocol_sha256,
                  'task_eval': str(output / 'task_eval.json'),
                  'source_manifest': summary.get('evaluation_source_manifest')}
        result = self._publish(experiment, record)
        torch.cuda.empty_cache()
        return result


def run():
    """Use exactly the ordinary training CLI and lifecycle; keep the import lazy."""
    from .train import main as train_main
    return train_main()


def main():
    return run()


if __name__ == '__main__':
    main()
