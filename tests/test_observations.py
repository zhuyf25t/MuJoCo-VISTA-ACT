"""Continuous VISTA sampling and unique persisted ownership of control frames."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.messages import AIMessage

import config
from observations import FrameObservation, ObservationWindow
from brains.vista import VistaBrain
from brains.vista.settings import VistaSettings
from brains.vista.storage import VistaStorage
from control.tools import ToolLayer
from sim.env import MobileManipEnv


def frame(number):
    return FrameObservation(number, np.full((4, 6, 3), number, np.uint8), (0.1 * number, 0.2, 1., 0., 0.))


@pytest.fixture
def storage(tmp_path):
    value = VistaStorage(VistaSettings(max_model_calls=8, output_root=tmp_path / "vista"))
    value.begin_episode(tmp_path / "raw", "test")
    return value


def read_records(storage):
    return [json.loads(line) for line in (storage.episode_dir / "actions.jsonl").read_text(encoding="utf-8").splitlines()]


def complete(storage, start, end):
    identifier = storage.begin_batch(start)
    storage.bind_batch_call(identifier, f"call-{identifier}", [{"action": "observe", "args": {}}])
    storage.archive_frames(tuple(frame(i) for i in range(start, end + 1)))
    storage.complete_batch(identifier, end, [{"action": "observe", "args": {}, "status": "completed",
                                            "frame_range": [start, end]}], "completed")


def test_frame_owns_readonly_pixels_and_immutable_state():
    pixels = np.zeros((2, 3, 3), np.uint8)
    state = [0., 0., 1., 0., 0.]
    observed = FrameObservation(0, pixels, state)
    pixels[:] = 10
    state[0] = 10
    assert not observed.image.any() and observed.state[0] == 0
    with pytest.raises(ValueError):
        observed.image.setflags(write=True)
    with pytest.raises(ValueError, match="不连续"):
        ObservationWindow((frame(0), frame(2)))


def test_env_reads_do_not_render_or_consume_and_ack_keeps_boundary():
    env = MobileManipEnv.__new__(MobileManipEnv)
    env._observation_stream_active = False
    env._observation_buffer = []
    env._observation_time = None
    now, captures = [4.0], []
    env.hal = SimpleNamespace(now=lambda: now[0])

    def capture(number):
        captures.append((number, now[0]))
        return frame(number)

    env._capture_frame = capture
    env.start_observation_stream()
    assert env.get_obs().current.frame_no == 0
    for _ in range(3):
        now[0] += config.CTRL_DT
        env.capture_observation_tick()
    assert [f.frame_no for f in env.get_obs().frames] == [0, 1, 2, 3]
    assert env.get_obs().current is env.get_obs().current
    assert len(captures) == 4
    env.ack_observations(3)
    now[0] += config.CTRL_DT
    env.capture_observation_tick()
    assert [f.frame_no for f in env.get_obs().frames] == [3, 4]
    with pytest.raises(RuntimeError, match="完整控制拍"):
        env.capture_observation_tick()
    env.stop_observation_stream()
    assert env.get_obs().current.frame_no == 4
    with pytest.raises(RuntimeError, match="未开始"):
        env.capture_observation_tick()


def test_batches_and_terminal_own_each_frame_once(storage):
    storage.ensure_frame(frame(0))
    complete(storage, 0, 3)
    complete(storage, 3, 5)
    storage.ensure_terminal_observation("decision_limit")
    storage.ensure_terminal_observation("decision_limit")
    rows = read_records(storage)
    assert [row["frame_range"] for row in rows] == [[0, 3], [3, 5], [5, 5]]
    assert [[f["frame_id"] for f in row["observations"]] for row in rows] == [
        ["f000000", "f000001", "f000002"], ["f000003", "f000004"], ["f000005"]]
    assert len(list((storage.episode_dir / "frames").glob("*.png"))) == 6
    assert not (storage.episode_dir / "frame_index.jsonl").exists()
    assert np.asarray(storage.load_image("f000002"))[0, 0, 0] == 2
    encoded = json.dumps(rows)
    assert "timestamp_s" not in encoded and "result_obs" not in encoded and "expectation" not in encoded


def test_initial_finish_keeps_o0(storage):
    storage.ensure_frame(frame(0))
    identifier = storage.begin_batch(0)
    assert read_records(storage)[0]["execution_status"] == "pending"
    storage.bind_batch_call(identifier, "finish", [])
    storage.finish_batch(identifier)
    storage.ensure_terminal_observation()
    row, = read_records(storage)
    assert row["execution_status"] == "finished" and row["observations"] == [frame(0).record()]


def test_zero_tick_recovery_transfers_unique_current_observation(storage):
    storage.ensure_frame(frame(0))
    complete(storage, 0, 0)
    complete(storage, 0, 2)
    storage.ensure_terminal_observation()
    rows = read_records(storage)
    assert rows[0]["observations"] == [] and rows[0]["frame_range"] == [0, 0]
    assert [obs["frame_id"] for row in rows for obs in row["observations"]] == [
        "f000000", "f000001", "f000002"]


def test_zero_tick_terminal_does_not_duplicate_o0(storage):
    storage.ensure_frame(frame(0))
    complete(storage, 0, 0)
    storage.ensure_terminal_observation()
    row, = read_records(storage)
    assert row["observations"] == [frame(0).record()]


def test_atomic_record_failure_does_not_advance_memory(storage, monkeypatch):
    import brains.vista.storage as module
    storage.ensure_frame(frame(0))
    identifier = storage.begin_batch(0)
    previous = (storage.episode_dir / "actions.jsonl").read_bytes()

    def fail(*args):
        raise OSError("disk failure")

    with monkeypatch.context() as patch:
        patch.setattr(module, "_atomic_text", fail)
        with pytest.raises(RuntimeError, match="STORAGE_IO_ERROR"):
            storage.bind_batch_call(identifier, "call", [{"action": "observe", "args": {}}])
    assert (storage.episode_dir / "actions.jsonl").read_bytes() == previous
    assert storage.history()["batches"][0]["tool_call_id"] is None
    storage.bind_batch_call(identifier, "call", [{"action": "observe", "args": {}}])


def test_abort_keeps_complete_samples_and_unknown_execution(storage):
    storage.ensure_frame(frame(0))
    identifier = storage.begin_batch(0)
    storage.bind_batch_call(identifier, "call", [{"action": "observe", "args": {}}])
    storage.archive_frames((frame(0), frame(1), frame(2)))
    storage.abort_batch()
    storage.ensure_terminal_observation("interrupted")
    rows = read_records(storage)
    assert rows[0]["execution_status"] == rows[1]["execution_status"] == "interrupted"
    assert rows[0]["actions"][0]["action_applied"] is None
    assert [obs["frame_id"] for row in rows for obs in row["observations"]] == [
        "f000000", "f000001", "f000002"]


def test_archive_rejects_gap_or_modified_boundary(storage):
    storage.ensure_frame(frame(0))
    with pytest.raises(RuntimeError, match="不连续"):
        storage.ensure_frame(frame(2))
    with pytest.raises(RuntimeError, match="发生变化"):
        storage.ensure_frame(FrameObservation(0, frame(1).image, frame(0).state))
    assert list(storage.frames) == ["f000000"]


def test_render_failure_does_not_relabel_a_completed_tick_as_zero_ticks():
    env = MobileManipEnv.__new__(MobileManipEnv)
    env._observation_stream_active = False
    env._observation_buffer = []
    env._observation_time = None
    now = [0.0]
    env.hal = SimpleNamespace(now=lambda: now[0])

    def capture(number):
        if number:
            raise RuntimeError("renderer unavailable")
        return frame(number)

    env._capture_frame = capture
    env.start_observation_stream()
    now[0] = config.CTRL_DT
    with pytest.raises(RuntimeError, match="renderer unavailable"):
        env.capture_observation_tick()
    assert env.current_frame_no == 1
    assert env.get_obs().current.frame_no == 0
    with pytest.raises(RuntimeError, match="不能跳过缺帧"):
        env.capture_observation_tick()


def test_partial_archive_recovery_preserves_execution_and_unique_frames(storage):
    storage.ensure_frame(frame(0))
    identifier = storage.begin_batch(0)
    storage.bind_batch_call(identifier, "call", [{"action": "observe", "args": {}}])
    storage.ensure_frame(frame(1))
    actions = [{"action": "observe", "args": {}, "status": "completed", "frame_range": [0, 3],
                "action_applied": True, "error": None}]
    storage.complete_batch(identifier, 3, actions, "completed",
                           observation_error={"code": "FRAME_UNAVAILABLE", "message": "PNG write failed"})
    row, = read_records(storage)
    assert row["frame_range"] == [0, 3] and row["actions"] == actions
    assert [obs["frame_id"] for obs in row["observations"]] == ["f000000", "f000001"]
    storage.archive_frames(tuple(frame(i) for i in range(4)))
    storage.abort_batch()
    storage.recover_observations()
    storage.ensure_terminal_observation("interrupted")
    rows = read_records(storage)
    assert rows[0]["actions"] == actions and rows[0]["execution_status"] == "completed"
    assert rows[0]["observation_error"]["recovered"] is True
    assert [obs["frame_id"] for row in rows for obs in row["observations"]] == [
        "f000000", "f000001", "f000002", "f000003"]


def test_real_collection_preserves_act_pre_action_pairing_and_excludes_tail(tmp_path):
    from run_collect import run_episode

    class Model:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages, **kwargs):
            self.calls += 1
            name = "play" if self.calls == 1 else "finish"
            args = ({"actions": [{"action": "open_gripper", "args": {}},
                                 {"action": "forward", "args": {"seconds": 0.2}},
                                 {"action": "shoulder", "args": {"delta": 0.1}}], "expectation": "test",
                     "basis_frame_id": "f000000"} if self.calls == 1 else
                    {"success": False, "reason": "test finished"})
            return AIMessage(content="", tool_calls=[{"id": name, "name": name, "args": args}])

    model = Model()
    brain = VistaBrain(settings=VistaSettings(max_model_calls=8, output_root=tmp_path / "vista"), model=model)
    schemas = ToolLayer(None, None, None).schemas_anthropic()
    try:
        with MobileManipEnv() as env:
            env.reset(seed=17)
            outcome = run_episode(env, brain, schemas, "test", tmp_path / "raw", save_images=True, verbose=False)
            assert not env.observation_stream_active
        rows = read_records(brain.storage)
        end = rows[0]["frame_range"][1]
        assert [row["frame_range"] for row in rows] == [[0, end], [end, end]]
        spans = [action["frame_range"] for action in rows[0]["actions"]]
        assert spans[0] == [0, 1]  # Already open: one tick, formerly eight.
        assert spans[1][0] == 1 and spans[2] == [spans[1][1], end]
        assert 3 <= spans[1][1] - 1 <= 4 and 2 <= end - spans[2][0] <= 4
        assert len(brain.storage.frames) == end + 1 and model.calls == 2
        raw = [json.loads(line) for line in (tmp_path / "raw" / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(raw) == end + round(config.EPISODE_SETTLE_SECONDS / config.CTRL_DT)
        for obs, step in zip(rows[0]["observations"], raw):
            np.testing.assert_allclose(obs["state"], step["state"], atol=1e-12)
            assert step["t_end"] - step["t"] == pytest.approx(config.CTRL_DT)
            from PIL import Image
            with Image.open(tmp_path / "raw" / step["imgs"][config.CAM_FRONT]) as image:
                np.testing.assert_array_equal(np.asarray(brain.storage.load_image(obs["frame_id"])), np.asarray(image))
        assert any(abs(step["action"][3]) > 0 for step in raw[:end])
        assert raw[end - 1]["state"][0] != raw[0]["state"][0]
        assert outcome["stop_code"] == "brain_done"
        metadata = json.loads((brain.storage.episode_dir / "outcome.json").read_text(encoding="utf-8"))
        assert metadata["format_version"] == 2 and metadata["fps"] == 10
        assert metadata["collection_dir"] == str((tmp_path / "raw").resolve())
    finally:
        brain.close()


def test_real_render_failure_preserves_actual_tick_range_without_fabricating_pixels(tmp_path, monkeypatch):
    from run_collect import run_episode

    class Model:
        def invoke(self, messages, **kwargs):
            return AIMessage(content="", tool_calls=[{"id": "move", "name": "play", "args": {
                "actions": [{"action": "forward", "args": {"seconds": 0.3}}],
                "expectation": "test", "basis_frame_id": "f000000"}}])

    brain = VistaBrain(settings=VistaSettings(max_model_calls=8, output_root=tmp_path / "vista"), model=Model())
    try:
        with MobileManipEnv() as env:
            env.reset(seed=17)
            original = env._capture_frame

            def capture(number):
                if number == 2:
                    raise RuntimeError("injected renderer failure")
                return original(number)

            monkeypatch.setattr(env, "_capture_frame", capture)
            with pytest.raises(RuntimeError, match="执行进度未知"):
                run_episode(env, brain, ToolLayer(None, None, None).schemas_anthropic(), "test",
                            tmp_path / "raw", save_images=False, verbose=False)
            assert env.current_frame_no == 2 and env.get_obs().current.frame_no == 1
            assert not env.observation_stream_active
        row, = read_records(brain.storage)
        assert row["frame_range"] == row["actions"][0]["frame_range"] == [0, 2]
        assert row["actions"][0]["status"] == "unknown"
        assert row["observation_error"]["code"] == "FRAME_UNAVAILABLE"
        assert [obs["frame_id"] for obs in row["observations"]] == ["f000000", "f000001"]
        assert not (brain.storage.episode_dir / "frames" / "f000002.png").exists()
        assert len((tmp_path / "raw" / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()) == 2
        assert "injected renderer failure" in (tmp_path / "raw" / "decisions.jsonl").read_text(encoding="utf-8")
    finally:
        brain.close()
