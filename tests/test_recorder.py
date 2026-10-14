"""ACT alignment, append-only dataset identity, and real process-crash recovery."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import config
from recorder import DatasetStore, EpisodeRecorder, next_episode_dir


def environment():
    data = SimpleNamespace(time=0.0, qpos=np.zeros(2), qvel=np.zeros(2), ctrl=np.zeros(5))
    robot = SimpleNamespace(arm_q=np.array([0.1, -0.2]), finger_opening=0.4,
                            base_velocity=np.array([0.02, -0.03]), base_pose=np.zeros(3),
                            hal=SimpleNamespace(applied_action=np.array([0.5, -0.8, 0, 0.3, 0])))

    def capture(data):
        return {config.CAM_FRONT: np.full((240, 320, 3), round(data.time * 10), np.uint8)}

    return data, robot, capture


def tick(recorder, data, robot):
    recorder.before_tick(None, data, robot)
    data.time += config.CTRL_DT
    robot.arm_q += 0.01
    data.qpos += 0.01
    recorder.on_tick(None, data, robot)


def entries(store):
    return json.loads(store.manifest_path.read_text(encoding="utf-8"))["episodes"]


def test_success_exports_exact_pre_action_pixels_state_targets_and_all_frames(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    data, robot, capture = environment()
    source = tmp_path / "raw/ep_0000"
    with EpisodeRecorder(source, {"seed": 4}, capture, store) as rec:
        assert entries(store)[0]["selected_for_dataset"] is False
        expected_states = []
        expected_actions = []
        for i in range(3):
            expected_states.append([*robot.arm_q, robot.finger_opening, *robot.base_velocity])
            robot.hal.applied_action = np.array([0.5, -0.8, i % 2, 0.3 if i == 0 else 0, -0.1 * i])
            expected_actions.append(robot.hal.applied_action.copy())
            tick(rec, data, robot)
        rec.record_terminal(data, robot)
        result = rec.finish(True)
    assert result["num_steps"] == 3 and "split" not in result
    with h5py.File(store.root / result["h5"], "r") as f:
        paths = []
        f.visititems(lambda name, obj: paths.append(name) if isinstance(obj, h5py.Dataset) else None)
        assert sorted(paths) == ["action", "observations/images/front", "observations/state"]
        assert f["observations/images/front"].shape == (3, 240, 320, 3)
        assert f["observations/images/front"].dtype == np.uint8
        assert f["action"].dtype == f["observations/state"].dtype == np.float32
        np.testing.assert_array_equal(f["observations/images/front"][:, 0, 0, 0], [0, 1, 2])
        np.testing.assert_allclose(f["observations/state"][:], expected_states)
        np.testing.assert_allclose(f["action"][:], expected_actions)
        assert f.attrs["num_steps"] == 3
    assert json.loads((source / "terminal.json").read_text())["t"] == pytest.approx(0.3)
    assert len((source / "trajectory.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("success,info,status", [
    (False, {}, "rejected"),
    (True, {"stop_code": "window_closed"}, "interrupted"),
    (True, {"stop_code": "interrupted", "error": ""}, "interrupted"),
    (True, {"settling": {"completed": False}}, "interrupted"),
])
def test_failure_and_partial_episodes_keep_raw_but_never_publish(tmp_path, success, info, status):
    store = DatasetStore(tmp_path / "dataset")
    data, robot, capture = environment()
    with EpisodeRecorder(tmp_path / "raw", {}, capture, store) as rec:
        tick(rec, data, robot)
        result = rec.finish(success, info)
    assert result["status"] == status
    assert result["selected_for_dataset"] is False and "split" not in result
    assert not list((store.root / "episodes").glob("*.h5"))
    assert (tmp_path / "raw/trajectory.jsonl").stat().st_size > 0


def test_multiple_runs_and_raw_roots_append_without_reset_or_premature_splits(tmp_path):
    root = tmp_path / "dataset"
    for run in range(2):
        store = DatasetStore(root)
        assert store.recover() == []
        data, robot, capture = environment()
        with EpisodeRecorder(tmp_path / f"run_{run}/ep_0000", {}, capture, store) as rec:
            tick(rec, data, robot)
            rec.record_terminal(data, robot)
            rec.finish(True)
    rows = entries(store)
    assert [e["episode_id"] for e in rows] == ["episode-0000", "episode-0001"]
    assert all("split" not in e for e in rows)
    assert rows[0]["source_uid"] != rows[1]["source_uid"]


def test_active_episode_is_excluded_and_recovery_does_not_interrupt_it(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    with EpisodeRecorder(tmp_path / "raw", {}, dataset=store):
        assert store.recover() == []
        assert entries(store)[0]["status"] == "recording"
        assert entries(store)[0]["selected_for_dataset"] is False
    assert store.recover()[0]["status"] == "interrupted"
    assert store.recover() == []


CRASH_SCRIPT = """
import os, runpy, sys
from pathlib import Path
from recorder import DatasetStore, EpisodeRecorder
helpers = runpy.run_path('tests/test_recorder.py')
data, robot, capture = helpers['environment']()
store = DatasetStore(Path(sys.argv[1]))
rec = EpisodeRecorder(Path(sys.argv[2]), {}, capture, store)
helpers['tick'](rec, data, robot)
stage = sys.argv[3]
if stage == 'during_episode':
    os._exit(42)
rec.record_terminal(data, robot)
if stage == 'before_export':
    store._export = lambda *args: os._exit(42)
else:
    original = store._update
    def crash_after_export(episode_id, **values):
        if values.get('status') == 'ready':
            os._exit(42)
        return original(episode_id, **values)
    store._update = crash_after_export
rec.finish(True)
"""


@pytest.mark.parametrize("stage", ["during_episode", "before_export", "after_export"])
def test_hard_exit_without_finally_keeps_records_and_recovers_idempotently(tmp_path, stage):
    root, raw = tmp_path / "dataset", tmp_path / "raw"
    result = subprocess.run([sys.executable, "-B", "-c", CRASH_SCRIPT, str(root), str(raw), stage],
                            cwd=config.ROOT, capture_output=True, text=True, timeout=45)
    assert result.returncode == 42, result.stderr
    store = DatasetStore(root)
    before = entries(store)[0]
    assert before["selected_for_dataset"] is False and before["h5"] is None
    assert len((raw / "trajectory.jsonl").read_text().splitlines()) == 1
    recovered = store.recover()[0]
    if stage == "during_episode":
        assert recovered["status"] == "interrupted" and recovered["h5"] is None
    else:
        assert recovered["status"] == "ready" and recovered["num_steps"] == 1
        with h5py.File(root / recovered["h5"], "r") as f:
            assert f["action"].shape == (1, 5)
    assert store.recover() == [] and len(entries(store)) == 1


def test_concurrent_processes_cannot_lose_manifest_entries_or_reuse_raw_directories(tmp_path):
    script = """
import sys
from pathlib import Path
from recorder import DatasetStore, EpisodeRecorder, next_episode_dir
store = DatasetStore(Path(sys.argv[1]))
for i in range(3):
    with EpisodeRecorder(next_episode_dir(Path(sys.argv[2])), {}, dataset=store) as rec:
        rec.finish(False)
"""
    commands = [sys.executable, "-B", "-c", script, str(tmp_path / "dataset"), str(tmp_path / "raw")]
    processes = [subprocess.Popen(commands, cwd=config.ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True) for _ in range(3)]
    for process in processes:
        _, errors = process.communicate(timeout=45)
        assert process.returncode == 0, errors
    rows = entries(DatasetStore(tmp_path / "dataset"))
    assert len(rows) == len({e["episode_id"] for e in rows}) == len({e["source_dir"] for e in rows}) == 9
    assert all(e["status"] == "rejected" and not e["selected_for_dataset"] for e in rows)


@pytest.mark.parametrize("corruption", ["missing_frame", "wrong_state", "time_gap"])
def test_success_does_not_allow_corrupt_or_misaligned_data(tmp_path, corruption):
    store = DatasetStore(tmp_path / "dataset")
    data, robot, capture = environment()
    with EpisodeRecorder(tmp_path / "raw", {}, capture, store) as rec:
        tick(rec, data, robot)
        tick(rec, data, robot)
        rec.record_terminal(data, robot)
        path = rec.dir / "trajectory.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if corruption == "missing_frame":
            rows.pop()
        elif corruption == "wrong_state":
            rows[0]["state"] = [0, 1]
        else:
            rows[1]["t"] += 0.5
            rows[1]["t_end"] += 0.5
        rec._f_traj.close()
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        result = rec.finish(True)
    assert result["status"] == "invalid" and result["selected_for_dataset"] is False
    assert not list((store.root / "episodes").glob("*.h5"))


def test_recorder_refuses_post_action_only_and_partial_step(tmp_path):
    data, robot, capture = environment()
    with EpisodeRecorder(tmp_path / "raw", {}, capture) as rec:
        with pytest.raises(RuntimeError, match="缺少执行前"):
            rec.on_tick(None, data, robot)
        rec.before_tick(None, data, robot)
        data.time += 0.05
        with pytest.raises(ValueError, match="时长"):
            rec.on_tick(None, data, robot)
        rec.finish(False, {"stop_code": "interrupted"})
    assert (tmp_path / "raw/trajectory.jsonl").read_text() == ""


def test_mujoco_applied_action_uses_actuator_targets_and_clipping(tmp_path):
    from sim.env import MobileManipEnv
    from playback import load_trajectory
    with MobileManipEnv() as env:
        env.reset(seed=0)
        env.hal.command_arm(np.array([0.5, -0.8]))
        env.hal.command_finger(0.05)
        env.hal.command_drive(2, 1)  # Both wheels saturate at 5 rad/s.
        np.testing.assert_allclose(env.hal.applied_action, [0.5, -0.8, 0, 0.5, 0])
        before = env.robot.arm_q.copy()
        with EpisodeRecorder(tmp_path / "raw", {}, capture=None) as rec:
            env.hal.before_tick = lambda: rec.before_tick(env.model, env.data, env.robot)
            env.hal.tick(lambda: rec.on_tick(env.model, env.data, env.robot))
            env.hal.before_tick = None
            rec.record_terminal(env.data, env.robot)
            rec.finish(False)
        row = json.loads((tmp_path / "raw/trajectory.jsonl").read_text())
        np.testing.assert_allclose(row["state"][:2], before)
        np.testing.assert_allclose(row["action"], [0.5, -0.8, 0, 0.5, 0])
        frames = load_trajectory(tmp_path / "raw/trajectory.jsonl", env.model)
        assert len(frames) == 2 and frames[-1].t == env.data.time
