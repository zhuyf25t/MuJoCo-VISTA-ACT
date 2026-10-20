"""Train ACT on an explicitly prepared experiment; resume at epoch boundaries."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from recorder import FileLock
from .common import atomic_json, load_prepared
from .config import config_from_dict, load_config
from .dataset import EpisodeDataset
from .model import ACT
from .action_codec import CONTRACT
from .checkpoint import load_checkpoint, save_checkpoint, normalized_prediction
from .losses import hybrid_loss, training_class_balance, LossAverages


def masked_l1(prediction, target, is_pad):
    valid = (~is_pad).unsqueeze(-1)
    count = valid.sum() * prediction.shape[-1]
    if count.item() == 0:
        raise ValueError("A batch must contain valid actions")
    # torch.where also prevents arbitrary padded values from entering the loss.
    return torch.where(valid, (prediction - target).abs(), 0).sum() / count


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


class BatchProgress:
    """Live terminal bar, or a progress line every 10 seconds in redirected logs."""
    def __init__(self, total, description, mode):
        self.total, self.description, self.completed = total, description, 0
        self.started = self.last_report = time.monotonic()
        self.bar = tqdm(total=total, desc=description, unit="batch", file=sys.stdout,
                        disable={"auto": None, "always": False, "never": True}[mode],
                        dynamic_ncols=True, leave=True, position=0, mininterval=0.5,
                        miniters=1)
        self.log_progress = mode == "auto" and self.bar.disable
        if self.log_progress:
            print(f"{description}: 0/{total} batch，开始计算…", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.bar.close()

    def advance(self, **metrics):
        self.completed += 1
        postfix = ", ".join(f"{key}={'pending' if value is None else f'{value:.4f}'}"
                            for key, value in metrics.items())
        self.bar.set_postfix_str(postfix, refresh=False)
        self.bar.update(1)
        now = time.monotonic()
        if self.log_progress and (now - self.last_report >= 10 or self.completed == self.total):
            elapsed = now - self.started
            remaining = elapsed / self.completed * (self.total - self.completed)
            print(f"{self.description}: {self.completed}/{self.total} batch "
                  f"({self.completed / self.total:.0%}) | {elapsed:.0f}s, ETA {remaining:.0f}s | {postfix}",
                  flush=True)
            self.last_report = now


def print_epoch_summary(metrics, epochs, task_record):
    print(f"\n第 {metrics['epoch'] + 1}/{epochs} 轮 | {metrics['elapsed_seconds']:.1f} 秒 | "
          f"累计更新 {metrics['global_step']:,} 次", flush=True)
    print(f"  训练 loss={metrics['train_loss']:.4f} | 训练动作 L1={metrics['train_l1']:.4f}", flush=True)
    print(f"  闭环任务评分={task_record['score']} | 已保存 last.pt；best.pt 由闭环任务评估选择", flush=True)


def rng_state(generator):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0],
            torch.tensor(numpy_state[1].astype(np.int64)), *numpy_state[2:]],
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.numpy().astype(np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def provenance():
    try:
        repo = Path(__file__).resolve().parents[1]
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True))
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    import torchvision
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda,
            "git_revision": revision, "git_dirty": dirty}


def train(root, experiment, *, epochs, cfg=None, device="auto", resume=False, backbone_weights=None,
          progress="auto", workers=10, epoch_callback=None):
    """Train on every frozen demonstration and select best.pt by fresh task scenes."""
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive total epoch count")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if progress not in ("auto", "always", "never"):
        raise ValueError("progress must be auto, always or never")
    experiment = Path(experiment)
    if not (experiment / "prepared.json").is_file():
        raise ValueError("Run act.prepare first")
    with FileLock(experiment / ".training.lock", timeout=0.2):
        return _train(root, experiment, epochs, cfg, device, resume, backbone_weights, progress,
                      workers, epoch_callback)


def _train(root, experiment, epochs, cfg, device, resume, backbone_weights, progress,
           workers=10, epoch_callback=None):
    from .checkpoint import DEFAULT_DEPLOYMENT
    from .experiment import EpochEvaluator, archive_source

    last = experiment / "last.pt"
    checkpoint = load_checkpoint(last) if resume else None
    if resume:
        if cfg is not None and cfg.to_dict() != config_from_dict(checkpoint["config"]).to_dict():
            raise ValueError("Resume configuration differs from checkpoint; start a new experiment")
        if backbone_weights:
            raise ValueError("Resume restores all weights; do not pass ImageNet weights")
        cfg = config_from_dict(checkpoint["config"])
        if (checkpoint.get("selection_metric") != "closed_loop_task_v2"
                or checkpoint.get("deployment") != DEFAULT_DEPLOYMENT):
            raise ValueError("Resume requires an all-demonstration task-selected checkpoint")
    else:
        if any((experiment / name).exists() for name in ("last.pt", "best.pt", "config.yaml", "metrics.jsonl")):
            raise FileExistsError("Experiment already started; use --resume or a new experiment directory")
        cfg = cfg or load_config()
    if cfg.model.chunk_size != 32:
        raise ValueError("Training protocol requires chunk_size=32")
    if cfg.training.patience or cfg.training.min_delta:
        raise ValueError("All-demonstration training has no early stopping or loss-based selection")
    print("正在核对全部示范清单和 HDF5 哈希…", flush=True)
    manifest, splits, stats, prepared = load_prepared(experiment, root, cfg.model)
    if (prepared.get("format_version") != 2 or prepared.get("training_mode") != "all_demonstrations"
            or splits["val"] or set(splits["train"]) != {e["episode_id"] for e in manifest["episodes"]}):
        raise ValueError("Training requires prepared v2/all_demonstrations with every episode in train and val=[]")
    evaluation_seeds = tuple(range(520, 540))
    if {e["seed"] for e in manifest["episodes"]} & set(evaluation_seeds):
        raise ValueError("Training scene seeds overlap closed-loop evaluation seeds 520..539")
    if resume and checkpoint["prepared"] != prepared:
        raise ValueError("Resume dataset differs from checkpoint")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu, cuda, cuda:N, auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    generator = torch.Generator().manual_seed(cfg.training.seed)
    dataset = EpisodeDataset(root, manifest["episodes"], stats, cfg.model)
    balance = training_class_balance(dataset, cfg.training.balance_power, cfg.training.first_query_weight)
    if resume and checkpoint["class_balance"] != balance:
        raise ValueError("Resume class statistics differ from checkpoint")
    atomic_json(experiment / "class_balance.json", balance)
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True,
                        num_workers=cfg.training.num_workers, drop_last=False,
                        pin_memory=device.type == "cuda", generator=generator)
    print("正在初始化 ACT…", flush=True)
    model = ACT(cfg.model, initialize_backbone=not resume, backbone_weights=backbone_weights).to(device)
    backbone, other = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith("backbone.") else other).append(parameter)
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": cfg.training.backbone_learning_rate},
                                   {"params": other, "lr": cfg.training.learning_rate}], weight_decay=cfg.training.weight_decay)
    amp = cfg.training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch = step = 0
    if resume:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, step = checkpoint["epoch"] + 1, checkpoint["global_step"]
        restore_rng(checkpoint["rng"], generator)
        # last.pt is authoritative if interruption occurred before the journal append.
        log_path = experiment / "metrics.jsonl"
        rows = []
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row["epoch"] < start_epoch:
                    rows.append(row)
        if not rows or rows[-1]["epoch"] != checkpoint["epoch"]:
            rows.append(checkpoint["metrics"])
        log_path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    else:
        (experiment / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    saved_rng = rng_state(generator)
    try:
        evaluator = EpochEvaluator(evaluation_seeds, device=str(device), workers=workers, video_every=0)
        evaluator.freeze_protocol(experiment)
        evaluator.source_manifest = archive_source(experiment)
    finally:
        restore_rng(saved_rng, generator)

    def evaluate_saved(saved):
        # Disk-loaded evaluation and optional observers cannot change subsequent batches.
        state = rng_state(generator)
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            record = evaluator(experiment, saved)
            if epoch_callback is not None:
                epoch_callback(experiment, saved)
            return record
        finally:
            restore_rng(state, generator)

    versions = provenance()
    atomic_json(experiment / "runtime.json", {**versions, "device": str(device), "amp_enabled": amp,
                "target_epochs": epochs, "resumed": resume, "sample_counts": prepared["sample_counts"],
                "updates_per_epoch": len(loader), "training_mode": "all_demonstrations",
                "evaluation_seeds": list(evaluation_seeds), "evaluation_workers": workers,
                "selection_metric": "closed_loop_task_v2", "deployment": dict(DEFAULT_DEPLOYMENT)})
    print(f"训练全部 {len(manifest['episodes'])} 条示范 / {len(dataset):,} 帧 | K=32 | "
          f"每轮 {len(loader):,} 次更新；每轮以 520..539 场景闭环评估选择 best.pt", flush=True)
    if resume:
        evaluate_saved(checkpoint)  # Repair an interrupted evaluation even if epochs are already complete.
        if start_epoch >= epochs:
            return experiment / "best.pt"
    for epoch in range(start_epoch, epochs):
        started = time.monotonic()
        model.train()
        action_total = action_count = 0.0
        losses = LossAverages()
        with BatchProgress(len(loader), f"Train {epoch + 1:02d}/{epochs}", progress) as batches:
            for batch in loader:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else nullcontext()
                with context:
                    output = model(batch["images"], batch["state"])
                    loss, parts, counts = hybrid_loss(output, batch["actions"], batch["is_pad"],
                                                       stats, balance, batch["action_states"], cfg.training.first_query_weight)
                with torch.no_grad():
                    prediction = normalized_prediction(output, stats)
                    l1 = masked_l1(prediction, batch["actions"], batch["is_pad"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                step += 1
                valid_elements = counts["valid"] * cfg.model.action_dim
                action_total += float(l1) * valid_elements
                action_count += valid_elements
                losses.update(parts, counts)
                batches.advance(train_loss=losses.result("train")["train_loss"], train_L1=action_total / action_count)
        # Evaluation workers share the GPU; release only transient training storage.
        optimizer.zero_grad(set_to_none=True)
        del batch, output, loss, parts, prediction, l1, context
        metrics = {"epoch": epoch, "global_step": step, "train_l1": action_total / action_count,
                   **losses.result("train"), "elapsed_seconds": time.monotonic() - started}
        checkpoint = {"format_version": 2, "action_codec": CONTRACT, "class_balance": balance,
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict(), "config": cfg.to_dict(), "schema": manifest["schema"],
                      "stats": stats, "prepared": prepared, "epoch": epoch, "global_step": step,
                      "selection_metric": "closed_loop_task_v2", "deployment": dict(DEFAULT_DEPLOYMENT),
                      "rng": rng_state(generator), "metrics": metrics, "versions": versions}
        save_checkpoint(last, checkpoint)
        with (experiment / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            stream.flush()
        task_record = evaluate_saved(checkpoint)
        print_epoch_summary(metrics, epochs, task_record)
    print(f"训练结束。闭环任务最佳模型：{experiment / 'best.pt'}；完整续训状态：{last}", flush=True)
    return experiment / "best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config")
    parser.add_argument("--epochs", type=int, required=True, help="Total epochs, including previously completed epochs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=10, help="Parallel closed-loop evaluation workers")
    parser.add_argument("--resume", action="store_true", help="Restore last.pt; no ImageNet download")
    parser.add_argument("--backbone-weights", help="Optional local official torchvision ResNet-18 state_dict")
    parser.add_argument("--progress", choices=("auto", "always", "never"), default="auto",
                        help="auto: terminal bars or 10-second log updates; always: force bars; never: summaries only")
    args = parser.parse_args()
    train(args.dataset_root, args.experiment, epochs=args.epochs, cfg=load_config(args.config) if args.config else None,
          device=args.device, resume=args.resume, backbone_weights=args.backbone_weights, progress=args.progress, workers=args.workers)


if __name__ == "__main__":
    main()
