"""Real-physics preset calibration; known fixtures are never exposed to a brain."""
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from control.drivers import GraspNotConfirmed
from sim.env import MobileManipEnv


FINGERS = ("arm_left_finger_geom", "arm_right_finger_geom")
# Stable bilateral acquisition changes the actual grip position. Edge fixtures
# measure 39.55+ mm during transport: nominal 4 cm with 1 mm pose tolerance.
MIN_RIM_CLEARANCE = 0.039


def place_pickup_fixture(env, dx=0.0, dy=0.0):
    """Ground ball at the calibrated pickup distance; other balls off the path."""
    bx, by, _ = env.hal.base_pose
    positions = [(bx + 0.545 + dx, by + dy), (0.2, 1.2), (0.6, 1.2), (1.0, 1.2)]
    for name, (x, y) in zip(config.BALL_NAMES, positions):
        joint = env.model.joint(f"{name}_joint")
        qa, va = joint.qposadr[0], joint.dofadr[0]
        env.data.qpos[qa:qa + 3] = [x, y, config.BALL_R]
        env.data.qvel[va:va + 6] = 0
    mujoco.mj_forward(env.model, env.data)
    env.hal.step_ticks(5)


def sphere_image_bounds(env):
    """Projected ball surface; bounds do not measure occlusion by the gripper."""
    mujoco.mj_forward(env.model, env.data)
    azimuth, elevation = np.meshgrid(np.linspace(0, 2 * np.pi, 81),
                                     np.linspace(0, np.pi, 41))
    surface = np.column_stack((np.sin(elevation).ravel() * np.cos(azimuth).ravel(),
                               np.sin(elevation).ravel() * np.sin(azimuth).ravel(),
                               np.cos(elevation).ravel()))
    points = env.data.body("ball_0").xpos + config.BALL_R * surface
    cid = env.model.camera(config.CAM_FRONT).id
    camera = (points - env.data.cam_xpos[cid]) @ env.data.cam_xmat[cid].reshape(3, 3)
    assert np.all(camera[:, 2] < 0), "ball must be in front of camera"
    w, h = config.CAM_FRONT_RES
    focal = h / (2 * np.tan(np.deg2rad(env.model.cam_fovy[cid] / 2)))
    pixels = np.column_stack((w / 2 + focal * camera[:, 0] / -camera[:, 2],
                              h / 2 - focal * camera[:, 1] / -camera[:, 2]))
    return pixels.min(axis=0), pixels.max(axis=0)


def assert_visible_ball(env):
    lo, hi = sphere_image_bounds(env)
    width, height = config.CAM_FRONT_RES
    # Higher transport poses may clip the sphere's top, but must keep most of
    # its vertical projection and its full width available for visual checks.
    assert lo[0] > 3 and hi[0] < width - 3
    assert 25 < hi[1] < height - 3
    visible_height = hi[1] - max(0, lo[1])
    assert visible_height / (hi[1] - lo[1]) >= 0.65


def drive_to_x(env, target):
    """Fixture positioning through the real wheel controller, not an agent policy."""
    for _ in range(12):
        distance = target - env.hal.base_pose[0]
        if abs(distance) < 0.012:
            return
        seconds = min(3.0, max(0.05, abs(distance) / config.PRIM_V))
        driver = env.robot.base.forward if distance > 0 else env.robot.base.back
        driver(seconds)
    pytest.fail("fixture base failed to reach release position")


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("dx,dy", [(0, 0), (0.02, -0.01), (0.02, 0.01)])
def test_ground_pickup_visible_carry_and_bin_release(seed, dx, dy, monkeypatch):
    with MobileManipEnv() as env:
        env.reset(seed=seed)
        place_pickup_fixture(env, dx, dy)
        fingers = {env.model.geom(n).id for n in FINGERS}
        floor = env.model.geom("floor").id
        walls = {env.model.geom(f"bin_wall_{n}").id for n in ("xp", "xn", "yp", "yn")}
        ball = env.model.geom("ball_0_geom").id
        original = env.hal._auto_grasp
        check_clearance = False

        def check_contacts():
            # Called after every 2 ms physics step, including motion between presets.
            if check_clearance:
                assert env.hal.attached_object() == "ball_0", "lost grasp during transport"
                bottom = env.data.body("ball_0").xpos[2] - config.BALL_R
                assert bottom > config.BIN_WALL_TOP + MIN_RIM_CLEARANCE, "held ball lost rim clearance"
            for contact in env.data.contact:
                pair = {contact.geom1, contact.geom2}
                assert not (floor in pair and pair & fingers), "finger pressed into floor"
                assert not (pair & walls and pair & fingers), "finger hit bin rim"
                if env.hal.attached_object() is not None:
                    assert not (pair & walls and ball in pair), "carried ball hit bin rim"
            original()

        monkeypatch.setattr(env.hal, "_auto_grasp", check_contacts)
        env.robot.arm.move_pose("reach")
        start = env.hal.now()
        env.robot.arm.close_gripper()
        assert env.hal.now() - start == pytest.approx(1.2)
        assert env.hal.attached_object() == "ball_0"
        assert config.GRASP_OPENING_MIN < env.hal.finger_opening < config.GRASP_OPENING_MAX
        env.robot.arm.move_pose("carry")
        assert env.hal.attached_object() == "ball_0"
        assert env.data.body("ball_0").xpos[2] - config.BALL_R > config.BIN_WALL_TOP + MIN_RIM_CLEARANCE
        assert_visible_ball(env)
        check_clearance = True

        # Approach while keeping the carried ball outside the near rim.
        drive_to_x(env, 0.35)
        assert_visible_ball(env)
        carry_forward = env.hal.tcp_pos[0] - env.hal.base_pose[0]
        env.robot.arm.move_pose("drop")
        assert env.hal.tcp_pos[0] - env.hal.base_pose[0] > carry_forward + 0.04
        assert env.data.body("ball_0").xpos[2] - config.BALL_R > config.BIN_WALL_TOP + MIN_RIM_CLEARANCE
        assert_visible_ball(env)
        ball_forward = env.data.body("ball_0").xpos[0] - env.hal.base_pose[0]
        drive_to_x(env, config.BIN_POS[0] - ball_forward)
        assert_visible_ball(env)
        check_clearance = False  # Release must allow the ball to fall below the rim.
        env.robot.arm.open_gripper()
        env.hal.step_ticks(30)
        assert env.hal.attached_object() is None
        assert env.world.object_in_bin("ball_0")


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("dy", [-0.01, 0.01])
def test_edge_pickup_reports_unconfirmed_after_fixed_duration(seed, dy):
    with MobileManipEnv() as env:
        env.reset(seed=seed)
        place_pickup_fixture(env, -0.02, dy)
        env.robot.arm.move_pose("reach")
        start = env.hal.now()
        with pytest.raises(GraspNotConfirmed):
            env.robot.arm.close_gripper()
        assert env.hal.now() - start == pytest.approx(1.2)
        assert env.hal.attached_object() is None
        assert not env.hal._grasp_attempt_active
        env.hal.step_ticks(20)
        assert env.hal.attached_object() is None


@pytest.mark.parametrize("seed", [1, 2])
def test_reach_empty_claw_closes_without_floor_stall(seed):
    with MobileManipEnv() as env:
        env.reset(seed=seed)
        env.robot.arm.move_pose("reach")
        start = env.hal.now()
        with pytest.raises(GraspNotConfirmed):
            env.robot.arm.close_gripper()
        assert env.hal.now() - start == pytest.approx(1.2)
        assert env.hal.finger_opening < 0.05
        env.robot.arm.open_gripper()
        assert env.hal.finger_opening > 0.95
