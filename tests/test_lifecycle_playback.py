"""Regression coverage for episode isolation, cleanup, and state-based replay."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import config
from playback import Frame, PlaybackCursor, apply_frame, load_trajectory, trajectory_path
from sim.env import MobileManipEnv


@pytest.mark.parametrize("yaw", [0.0, np.pi / 2, -np.pi / 3])
def test_observation_measures_body_velocity_and_actual_joints_without_stepping(yaw, monkeypatch):
    with MobileManipEnv() as env:
        data, hal = env.data, env.hal
        data.qpos[hal.i_free + 3:hal.i_free + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        dadr = env.model.joint(config.JOINT_BASE).dofadr[0]
        # Motion contains a sideways component, which must not become forward speed.
        forward, sideways, turn = 0.23, -0.17, -0.31
        data.qvel[dadr:dadr + 6] = [forward * np.cos(yaw) - sideways * np.sin(yaw),
                                  forward * np.sin(yaw) + sideways * np.cos(yaw), 0, 0, 0, turn]
        data.qpos[[hal.j_sh, hal.j_el]] = [0.21, -0.32]
        data.qpos[env.model.joint(config.JOINT_FINGER).qposadr[0]] = 0.02
        hal.command_arm(np.array([0.8, 0.9]))
        hal.command_finger(config.FINGER_OPEN_CTRL)
        hal.command_drive(-0.4, 0.7)  # Goals intentionally disagree with actual motion.
        data.time = 1.25
        captures = []

        def capture():
            captures.append(data.time)
            return np.zeros((3, 4, 3), dtype=np.uint8)

        monkeypatch.setattr(hal, "front_image", capture)
        obs = env.get_obs()
        assert obs["observation"] == pytest.approx({
            "shoulder_rad": 0.21, "elbow_rad": -0.32, "gripper_opening": 0.6,
            "base_v_mps": forward, "base_w_radps": turn})
        assert captures == [1.25] and obs["timestamp_s"] == data.time == 1.25
        data.qvel[:] = 0
        assert hal.base_velocity == pytest.approx([0, 0])  # Drive commands alone are not feedback.


def test_camera_rig_closes_partial_construction(monkeypatch):
    from viz import CameraRig

    model = mujoco.MjModel.from_xml_path(str(config.SCENE_XML))
    closed = []
    monkeypatch.setattr(mujoco, "Renderer", lambda *a, **kw: SimpleNamespace(close=lambda: closed.append(True)))
    with pytest.raises(ValueError, match="相机不存在"):
        CameraRig(model, cams=[config.CAM_FRONT, "missing-camera"])
    assert closed == [True]


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_episode_closes_recording_resources_on_interruption(tmp_path, monkeypatch, failure):
    import run_collect
    from recorder import EpisodeRecorder

    closed = []
    recorders = []
    monkeypatch.setattr(mujoco, "Renderer", lambda *a, **kw: SimpleNamespace(close=lambda: closed.append(True)))

    def record(*args, **kwargs):
        recorder = EpisodeRecorder(*args, **kwargs)
        recorders.append(recorder)
        return recorder

    def fail():
        raise failure("interrupted")

    monkeypatch.setattr(run_collect, "EpisodeRecorder", record)
    with MobileManipEnv() as env:
        monkeypatch.setattr(env, "get_obs", fail)
        brain = SimpleNamespace(name="vista", begin_episode=lambda *a: None,
                                abort_episode=lambda *a: None)
        with pytest.raises(failure):
            run_collect.run_episode(env, brain, [], "test", tmp_path / "ep_test")
    assert len(closed) == len(config.REC_CAMS)
    assert recorders[0]._f_dec.closed and recorders[0]._f_traj.closed
    meta = json.loads((tmp_path / "ep_test/meta.json").read_text(encoding="utf-8"))
    assert meta["success"] is False and failure.__name__ in meta["reason"]


def test_main_closes_environment_when_brain_setup_fails(monkeypatch):
    import run_collect

    closed = []
    env = MobileManipEnv()
    env.hal._front_renderer = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(run_collect, "MobileManipEnv", lambda: env)

    def fail(*args):
        raise RuntimeError("brain setup failed")

    monkeypatch.setattr(run_collect, "build_brain", fail)
    with pytest.raises(RuntimeError, match="brain setup failed"):
        run_collect.main(["--episodes", "1"])
    assert closed == [True]
    env.close()  # Idempotent: no second renderer.close().
    assert closed == [True]


@pytest.fixture
def recorded_frames():
    with MobileManipEnv() as env:
        env.reset(seed=7)
        first = {"t": 10.0, "qpos": env.data.qpos.tolist(), "ctrl": env.data.ctrl.tolist()}
        env.data.qpos[env.hal.i_free] += 0.3
        second = {"t": 10.5, "qpos": env.data.qpos.tolist(), "ctrl": env.data.ctrl.tolist()}
        return env.model, [first, second]


def _write_frames(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_replay_restores_robot_and_ball_positions_without_physics(tmp_path, recorded_frames, monkeypatch):
    model, rows = recorded_frames
    path = tmp_path / "trajectory.jsonl"
    _write_frames(path, rows)
    frames = load_trajectory(path, model)
    data = mujoco.MjData(model)

    def forbidden(*args):
        pytest.fail("Historical replay must not advance physics")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    for frame, row in zip(frames, rows):
        apply_frame(model, data, frame)
        np.testing.assert_allclose(data.qpos, row["qpos"])
        np.testing.assert_allclose(data.ctrl, row["ctrl"])
        assert data.time == row["t"]
        for ball in config.BALL_NAMES:
            adr = model.joint(f"{ball}_joint").qposadr[0]
            np.testing.assert_allclose(data.body(ball).xpos, frame.qpos[adr:adr + 3])


@pytest.mark.parametrize("field,value", [
    ("qpos", [1, 2]), ("ctrl", []), ("t", 9), ("t", float("nan")),
])
def test_replay_validates_every_frame(tmp_path, recorded_frames, field, value):
    model, rows = recorded_frames
    rows[1][field] = value
    path = tmp_path / "trajectory.jsonl"
    _write_frames(path, rows)
    with pytest.raises(ValueError, match="第 2 行"):
        load_trajectory(path, model)


def test_replay_rejects_empty_or_decision_log(tmp_path, recorded_frames):
    model, _ = recorded_frames
    path = tmp_path / "trajectory.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="轨迹为空"):
        load_trajectory(path, model)
    _write_frames(path, [{"t": 0, "tool": "forward", "args": {}}])
    with pytest.raises(ValueError, match="qpos"):
        load_trajectory(path, model)


def test_playback_timing_pause_restart_and_end():
    frames = [Frame(t, np.zeros(1), np.zeros(1)) for t in [10, 10.5, 12]]
    cursor = PlaybackCursor(frames, speed=2)
    cursor.advance(0.3)
    assert cursor.index == 1
    cursor.toggle_pause()
    cursor.advance(30)
    assert cursor.index == 1
    cursor.toggle_pause()
    cursor.advance(1)
    assert cursor.index == 2 and cursor.finished and cursor.paused
    cursor.toggle_pause()
    assert cursor.index == 0 and not cursor.finished
    looping = PlaybackCursor(frames, loop=True)
    looping.advance(2.75)
    assert looping.index == 1 and not looping.finished


def test_replay_check_accepts_episode_or_file(tmp_path, recorded_frames, capsys):
    from replay_gui import main

    _, rows = recorded_frames
    episode = tmp_path / "ep_0000"
    episode.mkdir()
    path = episode / "trajectory.jsonl"
    _write_frames(path, rows)
    assert trajectory_path("ep_0000", tmp_path) == path
    assert trajectory_path(str(episode), tmp_path) == path
    assert main([str(path), "--check"]) == 0
    assert "校验通过" in capsys.readouterr().out


@pytest.mark.parametrize("speed", [0, -1, float("nan"), float("inf")])
def test_replay_rejects_invalid_speed(speed):
    with pytest.raises(ValueError, match="倍速"):
        PlaybackCursor([Frame(0, np.zeros(1), np.zeros(1))], speed=speed)
