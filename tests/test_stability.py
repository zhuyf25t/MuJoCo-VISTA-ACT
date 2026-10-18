"""Image comparison, bounded control waits and continuous sampling contracts."""

import numpy as np
import pytest

import config
from control.drivers import ActionFailed, ArmDriver, BaseDriver, GraspNotConfirmed
from control.stability import compare_images, compare_observations, wait_until_stable
from control.tools import ToolLayer
from observations import FrameObservation


def obs(number, state=(0., 0., 1., 0., 0.), image=None):
    return FrameObservation(number, np.zeros((240, 320, 3), np.uint8) if image is None else image, state)


def test_micro_pixel_change_is_tolerated_but_same_block_motion_does_not_cancel():
    a = np.zeros((240, 320, 3), np.uint8)
    b = a.copy()
    b[0, 0, 0] = 23  # A subpixel rasterization jump measured in the real scene.
    assert compare_images(a, b).stable
    a[:3, :1] = 255
    b = np.zeros_like(a)
    b[:3, 1:2] = 255
    assert np.array_equal(a[:3, :3].mean((0, 1)), b[:3, :3].mean((0, 1)))
    assert compare_images(a, b).changed_blocks == 1


@pytest.mark.parametrize("x,y", [(60, 60), (23, 23), (305, 225)])
def test_local_motion_is_rejected_even_with_more_than_98_percent_global_similarity(x, y):
    a = np.zeros((240, 320, 3), np.uint8)
    b = a.copy()
    a[y:y+12, x:x+12, 1] = 255
    b[y:y+12, x+3:x+15, 1] = 255
    result = compare_images(a, b)
    assert result.similarity > .98
    assert result.max_local_changed_ratio > .1 and not result.stable


def test_partial_edge_blocks_use_actual_pixel_count_and_uint8_does_not_wrap():
    a = np.full((4, 5, 3), 255, np.uint8)
    b = a.copy()
    b[3, 3:] = 0
    result = compare_images(a, b)
    assert result.total_blocks == 4 and result.changed_blocks == 1 and not result.stable
    assert compare_images(b, a) == result


@pytest.mark.parametrize("changed_blocks,stable", [(17, True), (18, False)])
def test_99point8_percent_threshold_limits_distributed_changes(changed_blocks, stable):
    a = np.zeros((240, 320, 3), np.uint8)
    b = a.copy()
    # Separated patches pass the local limit; only the global limit decides.
    for i in range(changed_blocks):
        y, x = (i // 10) * 30, (i % 10) * 30
        b[y:y+3, x:x+3] = 255
    result = compare_images(a, b)
    assert result.total_blocks == 8560 and result.changed_blocks == changed_blocks
    assert result.max_local_changed_ratio <= config.STABILITY_LOCAL_CHANGED_RATIO
    assert result.stable is stable


def test_local_guard_still_rejects_changes_above_99point8_percent_similarity():
    a = np.zeros((240, 320, 3), np.uint8)
    b = a.copy()
    b[60:63, 60:81] = 255
    result = compare_images(a, b)
    assert result.similarity > config.STABILITY_IMAGE_SIMILARITY
    assert result.max_local_changed_ratio > config.STABILITY_LOCAL_CHANGED_RATIO
    assert not result.stable


@pytest.mark.parametrize("image", [np.zeros((2, 2)), np.zeros((2, 2, 3)),
                                   np.zeros((0, 2, 3), np.uint8)])
def test_invalid_images_fail_instead_of_reporting_stable(image):
    with pytest.raises(ValueError):
        compare_images(image, image)


@pytest.mark.parametrize("index", range(5))
def test_each_state_dimension_can_veto_stability(index):
    state = np.array([0., 0., .5, 0., 0.])
    changed = state.copy()
    changed[index] += config.STABILITY_STATE_EPS[index] * 1.1
    assert not compare_observations(obs(0, state), obs(1, changed)).stable


def test_constant_speed_is_not_stationary_and_stale_frame_is_rejected():
    result = compare_observations(obs(0, (0, 0, 1, .1, 0)), obs(1, (0, 0, 1, .1, 0)))
    assert result.state_stable and not result.base_stopped and not result.stable
    with pytest.raises(ValueError, match="consecutive"):
        compare_observations(obs(1), obs(1))


class Hardware:
    def __init__(self):
        self.ticks = 0
        self.arm_q = np.zeros(2)
        self.arm_goal = self.arm_q.copy()
        self.finger_opening = 1.
        self.drive = (0., 0.)
        self.held = None
        self.attempt = False
        self.update = lambda: None
        self.cached = obs(0)
        self.commands = []

    def tick(self, on_tick=None):
        self.ticks += 1
        self.commands.append(self.drive)
        self.update()
        if on_tick:
            on_tick()

    def sample(self):
        self.cached = obs(self.ticks, (*self.arm_q, self.finger_opening, 0., 0.))

    def read(self):
        return self.cached

    def step_ticks(self, ticks, on_tick=None):
        for _ in range(ticks):
            self.tick(on_tick)

    def now(self):
        return self.ticks * config.CTRL_DT

    @property
    def base_pose(self):
        return np.zeros(3)

    def command_drive(self, v, w):
        self.drive = (v, w)

    def arm_geometry(self):
        return .38, .4, np.full(2, -3.), np.full(2, 3.)

    def command_arm(self, q):
        self.arm_goal = q.copy()

    def finger_ctrl_range(self):
        return 0., .05

    def command_finger(self, goal):
        self.attempt = goal >= config.FINGER_CLOSED_THRESHOLD

    def finish_grasp_attempt(self):
        self.attempt = False

    def attached_object(self):
        return self.held


def test_wait_at_least_one_tick_preserves_callback_and_never_exceeds_upper_bound():
    hardware = Hardware()
    result = wait_until_stable(hardware, .8, hardware.sample, hardware.read)
    assert result.reason == "stable" and result.ticks == hardware.ticks == 1
    result = wait_until_stable(hardware, .25, hardware.sample, hardware.read, ready=lambda: False)
    assert result.reason == "time_limit" and result.ticks == 2 and hardware.ticks == 3
    # No reader is a conservative, explicitly bounded standalone-control fallback.
    assert wait_until_stable(hardware, .3).ticks == 3


def test_missing_new_sample_and_callback_failure_do_not_become_success():
    hardware = Hardware()
    with pytest.raises(ValueError, match="consecutive"):
        wait_until_stable(hardware, .8, read_observation=hardware.read)
    def fail():
        raise OSError("recording failed")
    with pytest.raises(OSError, match="recording failed"):
        wait_until_stable(hardware, .8, fail, hardware.read)


def test_forward_motion_budget_is_not_shortened_by_stable_observations():
    hardware = Hardware()
    BaseDriver(hardware).forward(.3, hardware.sample, hardware.read)
    assert hardware.commands == [(config.PRIM_V, 0)] * 3 + [(0, 0)]


def test_blocked_arm_and_unconfirmed_grasp_cannot_use_static_images_to_succeed():
    hardware = Hardware()
    arm = ArmDriver(hardware)
    with pytest.raises(ActionFailed, match="臂被卡住"):
        arm.move_arm_q([1., 0.], hardware.sample, read_observation=hardware.read)
    # Four ramp ticks followed by all three allowed settling ticks.
    assert hardware.ticks == 7
    initial = hardware.ticks
    with pytest.raises(GraspNotConfirmed):
        arm.close_gripper(hardware.sample, hardware.read)
    assert hardware.ticks - initial == 12 and not hardware.attempt


def test_close_waits_for_confirmation_and_always_closes_the_attempt_window():
    hardware = Hardware()
    def update():
        if hardware.ticks >= 3:
            hardware.held = "confirmed"
    hardware.update = update
    assert ArmDriver(hardware).close_gripper(hardware.sample, hardware.read)[0]
    assert hardware.ticks == 3 and not hardware.attempt
    hardware = Hardware()
    def fail():
        raise RuntimeError("capture failed")
    with pytest.raises(RuntimeError):
        ArmDriver(hardware).close_gripper(fail, hardware.read)
    assert not hardware.attempt


def test_open_requires_reached_target_even_with_identical_observations():
    hardware = Hardware()
    hardware.finger_opening = .5
    hardware.sample()
    with pytest.raises(ActionFailed, match="张爪"):
        ArmDriver(hardware).open_gripper(hardware.sample, hardware.read)
    assert hardware.ticks == 8
    hardware.finger_opening = 1.
    hardware.sample()
    assert ArmDriver(hardware).open_gripper(hardware.sample, hardware.read)[0]
    assert hardware.ticks == 9


def test_observe_default_explicit_cap_and_invalid_args():
    hardware = Hardware()
    tools = ToolLayer(BaseDriver(hardware), ArmDriver(hardware), hardware,
                      on_tick=hardware.sample, read_observation=hardware.read)
    assert tools.execute("observe", {})[0] and hardware.ticks == 1
    assert tools.execute("observe", {"seconds": .8})[0] and hardware.ticks == 2
    for value in (0, -.1, True, float("nan"), 3.1):
        assert not tools.execute("observe", {"seconds": value})[0]
    assert hardware.ticks == 2


def test_real_scene_already_open_shortens_wait_and_raw_sampling_stays_contiguous():
    from sim.env import MobileManipEnv
    with MobileManipEnv() as env:
        env.reset(seed=17)
        env.start_observation_stream()
        tools = ToolLayer(env.robot.base, env.robot.arm, env.hal,
                          on_tick=env.capture_observation_tick,
                          read_observation=lambda: env.get_obs().current)
        assert tools.execute("open_gripper", {})[0]
        assert env.current_frame_no == 1  # Old implementation always took 8 ticks.
        assert tools.execute("observe", {"seconds": .8})[0]
        assert env.current_frame_no == 2
        assert [frame.frame_no for frame in env.get_obs().frames] == [0, 1, 2]


@pytest.mark.parametrize("seed", [17, 42])
def test_real_ball_motion_is_not_hidden_by_static_robot_and_background(seed):
    import mujoco
    from sim.env import MobileManipEnv
    with MobileManipEnv() as env:
        env.reset(seed=seed)
        # Fixture setup only: the detector receives RGB and five public values,
        # never object positions or velocity from the simulator.
        joint = env.model.joint("ball_0_joint")
        qa, va = joint.qposadr[0], joint.dofadr[0]
        env.data.qpos[qa:qa+3] = [.6, 0., config.BALL_R + .004]
        env.data.qvel[va:va+6] = [0., .05, 0., 0., 0., 0.]
        mujoco.mj_forward(env.model, env.data)
        env.start_observation_stream()
        previous = env.get_obs().current
        env.hal.tick(env.capture_observation_tick)
        result = compare_observations(previous, env.get_obs().current)
        assert result.state_stable and result.image.similarity > .98
        assert result.image.max_local_changed_ratio > .1 and not result.stable
