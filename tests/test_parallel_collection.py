"""Real spawn races plus collector directory/stop contract checks."""

import json
import multiprocessing
import time
from argparse import Namespace
from pathlib import Path

import pytest

from brains.vista.settings import VistaSettings
from brains.vista.storage import GuideConflict, VistaStorage
from run_collect import check_episode_paths, episode_path, run_parallel


def _guide_editor(root, index, barrier):
    storage = VistaStorage(VistaSettings(max_model_calls=12, output_root=Path(root)))
    storage.begin_episode(Path(root) / "raw" / f"ep_{index:04d}", "test",
                          episode_name=f"ep_{index:04d}")
    guide = storage.read_memory("guide")
    barrier.wait(timeout=30)  # All four deliberately draft from the same revision.
    for _ in range(10):
        try:
            storage.write_memory("guide", guide["content"] + f"\nexperience-{index}",
                                 base_revision=guide["revision"])
            return
        except GuideConflict:
            guide = storage.read_memory("guide")  # Simulate model reread + semantic merge.
    raise AssertionError("guide writer starved")


def test_four_process_guide_updates_preserve_all_contributions(tmp_path):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    processes = [context.Process(target=_guide_editor, args=(str(tmp_path), i, barrier)) for i in range(4)]
    try:
        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=45)
            assert p.exitcode == 0
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()
            p.close()
    guide = (tmp_path / "guide.md").read_text(encoding="utf-8")
    assert all(guide.count(f"experience-{i}") == 1 for i in range(4))
    audit = [json.loads(s) for s in (tmp_path / "guide_access.jsonl").read_text(encoding="utf-8").splitlines()]
    writes = [e for e in audit if e["operation"] == "write"]
    assert len(writes) == 4 and len({e["pid"] for e in writes}) == 4
    assert sum(e["operation"] == "conflict" for e in audit) >= 3
    assert all(writes[i]["base_revision"] == writes[i-1]["revision"] for i in range(1, 4))
    for e in writes:
        assert (tmp_path / "guide_versions" / f"{e['revision']}.md").is_file()


def test_seed_paths_do_not_follow_completion_order_and_reject_reuse(tmp_path):
    args = Namespace(seed=7, episodes=3, out=tmp_path / "raw", brain_memory=tmp_path / "vista")
    check_episode_paths(args)
    assert [episode_path(args.out, i).name for i in [9, 7, 8]] == ["ep_0009", "ep_0007", "ep_0008"]
    episode_path(args.brain_memory, 8).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="seed=8"):
        check_episode_paths(args)


def test_parallel_stop_file_prevents_any_dispatch(tmp_path):
    stop = tmp_path / "stop"
    stop.touch()
    class Dataset:
        recovered = False
        def recover(self):
            self.recovered = True
    dataset = Dataset()
    args = Namespace(seed=0, episodes=10, workers=10, stop_file=str(stop), out=tmp_path / "raw")
    assert run_parallel(args, dataset) == 1
    assert dataset.recovered and not args.out.exists()


def _stub_worker(args, seed, ep_dir):
    if args.stop_file:
        Path(args.stop_file).touch()
        time.sleep(0.3)
    elif seed == 1:
        raise RuntimeError("intentional worker crash")
    (ep_dir / "worker_result.json").write_text(json.dumps({"seed": seed, "success": True}), encoding="utf-8")


@pytest.mark.parametrize("stop", [True, False])
def test_spawn_dispatch_drains_on_stop_and_isolates_crashed_worker(tmp_path, monkeypatch, stop):
    import run_collect
    monkeypatch.setattr(run_collect, "_collect_worker", _stub_worker)
    class Dataset:
        def recover(self):
            return []
    args = Namespace(seed=0, episodes=4, workers=2, out=tmp_path / "raw",
                     stop_file=str(tmp_path / "STOP") if stop else None)
    assert run_parallel(args, Dataset()) == 1
    records = sorted(args.out.glob("ep_*/worker_result.json"))
    assert [p.parent.name for p in records] == (["ep_0000", "ep_0001"] if stop else
                                              ["ep_0000", "ep_0002", "ep_0003"])
