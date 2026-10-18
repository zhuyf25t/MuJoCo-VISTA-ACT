"""Grasp admission and lifecycle with bounded waits and real contacts."""
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import config
from control.drivers import GraspNotConfirmed
from control.tools import ToolLayer
from sim.backend import MjRobot
from sim.env import MobileManipEnv


STATES = json.loads((Path(__file__).parent / "fixtures/grasp_before_close.json")
                   .read_text(encoding="utf-8"))["states"]


def restore_before_close(env, seed):
    env.reset(seed=seed)  # Restore the recorded mass/friction randomization too.
    state = next(row for row in STATES if row["seed"] == seed)
    env.data.time = state["t"]
    for field in ("qpos", "qvel", "ctrl"):
        getattr(env.data, field)[:] = state[field]
    mujoco.mj_forward(env.model, env.data)


def assert_no_pending_grasp(hal):
    assert not hal._grasp_attempt_active
    assert hal._grasp_candidate is None
    assert not hal._grasp_samples


def test_observe_advances_eight_recorded_ticks_without_reopening_grasp():
    with MobileManipEnv() as env:
        restore_before_close(env, 1)
        with pytest.raises(GraspNotConfirmed):
            env.robot.arm.close_gripper()
        arm_target = env.data.ctrl[env.hal.i_act_arm].copy()
        finger_target = env.data.ctrl[env.hal.i_act_fi]
        env.hal.command_drive(0.2, 0.1)
        recorded = []
        layer = ToolLayer(env.robot.base, env.robot.arm, env.hal,
                          on_tick=lambda: recorded.append(env.hal.now()))
        start = env.hal.now()
        ok, _ = layer.execute("observe", {"seconds": 0.8})
        assert ok
        assert env.hal.now() - start == pytest.approx(0.8)
        assert np.array(recorded) - start == pytest.approx(np.arange(1, 9) * 0.1)
        np.testing.assert_array_equal(env.data.ctrl[env.hal.i_act_arm], arm_target)
        assert env.data.ctrl[env.hal.i_act_fi] == finger_target
        np.testing.assert_allclose(env.hal.applied_action[-2:], [0, 0])
        assert env.hal.attached_object() is None
        assert_no_pending_grasp(env.hal)


def test_unconfirmed_close_finishes_at_1point2_and_cannot_latch_later():
    with MobileManipEnv() as env:
        restore_before_close(env, 1)
        start = env.hal.now()
        with pytest.raises(GraspNotConfirmed, match="未确认稳定夹持"):
            env.robot.arm.close_gripper()
        assert env.hal.now() - start == pytest.approx(1.2)
        assert env.hal.attached_object() is None
        assert 0.4 < env.hal.finger_opening < 0.45  # Aperture alone is not confirmation.
        assert_no_pending_grasp(env.hal)

        env.hal.step_ticks(20)  # The servo target persists, but admission is closed.
        assert 0.28 < env.hal.finger_opening < 0.35
        assert "ball_1" in env.hal._valid_grasp_balls()
        assert env.hal.attached_object() is None

        start = env.hal.now()
        env.robot.arm.close_gripper()  # Only an explicit new action can acquire it.
        assert env.hal.now() - start == pytest.approx(1.2)
        assert env.hal.attached_object() == "ball_1"


def test_seed2_stable_bilateral_grasp_preserves_measured_aperture(monkeypatch):
    with MobileManipEnv() as env:
        restore_before_close(env, 2)
        events = []
        original = env.hal.attach_object

        def record_attachment(name):
            events.append((env.hal.now(), name, env.hal.finger_opening,
                           env.hal._valid_grasp_balls()))
            original(name)

        monkeypatch.setattr(env.hal, "attach_object", record_attachment)
        start = env.hal.now()
        env.robot.arm.close_gripper()
        assert env.hal.now() - start == pytest.approx(1.2)  # No early return on success.
        assert len(events) == 1
        when, name, opening, valid = events[0]
        assert config.GRASP_STABLE_SECONDS <= when - start <= 0.8
        assert name == "ball_2" and name in valid
        assert 0.28 < opening < 0.35
        assert env.hal.attached_object() == name
        assert_no_pending_grasp(env.hal)
        assert env.hal.applied_action[2] == 0  # Command stays fully closed.
        env.start_observation_stream()
        observation = env.get_obs().current.measurements()
        env.stop_observation_stream()
        assert 0.28 < observation["gripper_opening"] < 0.35
        assert "holding" not in observation

        for pose in ("carry", "drop"):
            env.robot.arm.move_pose(pose)
            assert env.hal.attached_object() == name
            assert 0.28 < env.hal.finger_opening < 0.35
            # Short drop trades 5 mm of height for a larger visible ball area.
            clearance = 0.03 if pose == "drop" else 0.04
            assert env.data.body(name).xpos[2] - config.BALL_R > config.BIN_WALL_TOP + clearance

        retained = []
        env.robot.arm.close_gripper(on_tick=lambda: retained.append(env.hal.attached_object()))
        assert retained == [name] * 12  # Repeated close never drops/reacquires the ball.
        assert len(events) == 1


@pytest.mark.parametrize("seed,confirmed", [(1, False), (2, True)])
def test_adaptive_close_requires_confirmation_before_early_return_and_keeps_carry(seed, confirmed):
    with MobileManipEnv() as env:
        restore_before_close(env, seed)
        env.start_observation_stream()
        kwargs = dict(on_tick=env.capture_observation_tick,
                      read_observation=lambda: env.get_obs().current)
        if confirmed:
            assert env.robot.arm.close_gripper(**kwargs)[0]
            assert 1 <= env.current_frame_no < 12
            held = env.hal.attached_object()
            assert held == "ball_2"
            assert_no_pending_grasp(env.hal)
            assert env.robot.arm.move_pose("carry", **kwargs)[0]
            assert env.hal.attached_object() == held
        else:
            with pytest.raises(GraspNotConfirmed):
                env.robot.arm.close_gripper(**kwargs)
            assert env.current_frame_no == 12 and env.hal.attached_object() is None
        assert_no_pending_grasp(env.hal)
        assert [frame.frame_no for frame in env.get_obs().frames] == list(range(env.current_frame_no + 1))


@pytest.mark.parametrize("phase", ["candidate", "held"])
def test_open_and_reset_clear_all_grasp_state(phase):
    with MobileManipEnv() as env:
        for ending in ("open", "reset"):
            restore_before_close(env, 2)
            env.hal.command_finger(config.FINGER_CLOSE_CTRL)
            env.hal.step_ticks(2 if phase == "candidate" else 8)
            if phase == "candidate":
                assert env.hal._grasp_candidate == "ball_2"
                assert env.hal.attached_object() is None
            else:
                assert env.hal.attached_object() == "ball_2"
            if ending == "open":
                before = env.data.qpos.copy()
                env.hal.command_finger(config.FINGER_OPEN_CTRL)
                np.testing.assert_array_equal(env.data.qpos, before)
            else:
                env.reset(seed=1)
            assert_no_pending_grasp(env.hal)
            assert env.hal.attached_object() is None
            assert env.hal._grasp_lost_since is None
            env.hal.step_ticks(10)
            assert env.hal.attached_object() is None


@pytest.mark.parametrize("callback", ["before_tick", "on_tick"])
@pytest.mark.parametrize("completed_ticks", [0, 8])
def test_interrupted_close_cannot_leave_an_acquisition_running(callback, completed_ticks):
    with MobileManipEnv() as env:
        restore_before_close(env, 1)
        calls = 0

        def interrupt():
            nonlocal calls
            calls += 1
            if calls > completed_ticks:
                raise RuntimeError("interrupted")

        kwargs = {}
        if callback == "before_tick":
            env.hal.before_tick = interrupt
        else:
            kwargs[callback] = interrupt
        with pytest.raises(RuntimeError, match="interrupted"):
            env.robot.arm.close_gripper(**kwargs)
        env.hal.before_tick = None
        assert_no_pending_grasp(env.hal)
        env.hal.step_ticks(20)
        assert env.hal.attached_object() is None


def test_grasp_lost_before_close_returns_is_a_failure(monkeypatch):
    with MobileManipEnv() as env:
        restore_before_close(env, 2)
        start = env.hal.now()
        valid = env.hal._valid_grasp_balls
        held = []
        monkeypatch.setattr(env.hal, "_valid_grasp_balls",
                            lambda: valid() if env.hal.now() - start < 0.8 else set())
        with pytest.raises(GraspNotConfirmed):
            env.robot.arm.close_gripper(on_tick=lambda: held.append(env.hal.attached_object()))
        assert env.hal.now() - start == pytest.approx(1.2)
        assert "ball_2" in held and held[-1] is None
        assert_no_pending_grasp(env.hal)


@pytest.mark.parametrize("brief", [True, False])
def test_contact_loss_debounces_then_releases_without_reacquiring(brief):
    with MobileManipEnv() as env:
        restore_before_close(env, 2)
        env.robot.arm.close_gripper()
        env.robot.arm.move_pose("carry")
        assert env.hal.attached_object() == "ball_2"
        m, d, hal = env.model, env.data, env.hal
        joint = m.joint(config.JOINT_FINGER)
        qa, va = joint.qposadr[0], joint.dofadr[0]
        original_q = float(d.qpos[qa])
        gain, bias = m.actuator_gainprm[hal.i_act_fi].copy(), m.actuator_biasprm[hal.i_act_fi].copy()
        # Simulate a physical finger fault, not an open command or a fake held flag.
        m.actuator_gainprm[hal.i_act_fi] = 0
        m.actuator_biasprm[hal.i_act_fi] = 0
        d.qpos[qa] = 0
        d.qvel[va] = 0
        hal.tick()
        assert hal.attached_object() == "ball_2"  # Less than a full loss window.
        if not brief:
            hal.tick()
            assert hal.attached_object() is None
            assert_no_pending_grasp(hal)
        m.actuator_gainprm[hal.i_act_fi] = gain
        m.actuator_biasprm[hal.i_act_fi] = bias
        if brief:
            d.qpos[qa] = original_q
            d.qvel[va] = 0
        hal.step_ticks(10)
        assert hal.attached_object() == ("ball_2" if brief else None)


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("surface,expected", [("inner", True), ("outer", False),
                                              ("tip", False), ("end_normal", False)])
def test_only_inward_finger_surfaces_qualify(side, surface, expected):
    with MobileManipEnv() as env:
        env.reset(seed=1)
        hal = env.hal
        finger = next(gid for gid, direction in hal._finger_geoms.items() if direction == side)
        half = env.model.geom_size[finger]
        point, normal = np.array([0., side * half[1], 0.]), np.array([0., 1., 0.])
        if surface == "outer":
            point[1] *= -1
        elif surface == "tip":
            point[2] = half[2] + 0.01
        elif surface == "end_normal":
            normal = np.array([0., 0., 1.])
        rotation = env.data.geom_xmat[finger].reshape(3, 3)
        contact = SimpleNamespace(pos=env.data.geom_xpos[finger] + rotation @ point,
                                  frame=np.r_[rotation @ normal, np.zeros(6)])
        assert hal._is_inner_pad_contact(contact, finger) is expected


@pytest.mark.parametrize("pairs,expected", [
    ([(1, 3), (1, 3)], set()),          # Two points on one finger.
    ([(1, 3), (2, 4)], set()),          # Different balls under the two fingers.
    ([(1, 3), (3, 2)], {"ball_0"}),    # Same ball, distinct fingers, reversed geom order.
])
def test_bilateral_contact_requires_distinct_fingers_on_the_same_ball(pairs, expected):
    # Contact grouping independent of collision multiplicity and geom ordering.
    fake = SimpleNamespace(finger_opening=0.32, _finger_geoms={1: -1, 2: 1},
                           _ball_geoms={3: "ball_0", 4: "ball_1"},
                           _ball_ids={"ball_0": 0, "ball_1": 1}, i_tcp=0,
                           _is_inner_pad_contact=lambda contact, finger: True,
                           data=SimpleNamespace(site_xpos=np.zeros((1, 3)), xpos=np.zeros((2, 3)),
                               contact=[SimpleNamespace(geom1=a, geom2=b, efc_address=0)
                                        for a, b in pairs]))
    assert MjRobot._valid_grasp_balls(fake) == expected
