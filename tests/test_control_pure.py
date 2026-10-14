"""纯控制层测试: 不 import mujoco —— 验证控制逻辑与仿真解耦.

test_kinematics: 两自由度解析 IK 往返一致性
test_base_driver_on_mock_hardware: 用理想 2D 单车模型当"假硬件",
跑真实 BaseDriver 控制循环, 证明控制代码可脱离 MuJoCo 运行.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control import ArmDriver, BaseDriver, PlanarArm2R  # noqa: E402
import config  # noqa: E402

import pytest  # noqa: E402


# ============================ 运动学 ============================
@pytest.mark.parametrize("df,dz", [(0.55, -0.19), (0.35, 0.10), (0.65, 0.05),
                                   (0.50, 0.20), (0.42, -0.12)])
def test_ik_fk_roundtrip(df, dz):
    arm = PlanarArm2R(0.38, 0.40, np.array([-0.3, -2.8]), np.array([2.2, 2.6]))
    q = arm.ik(df, dz)
    assert q is not None, f"可达目标无解: df={df}, dz={dz}"
    p = arm.fk(q)
    assert np.allclose(p, [df, dz], atol=1e-6)


def test_ik_out_of_reach():
    arm = PlanarArm2R(0.38, 0.40, np.array([-0.3, -2.8]), np.array([2.2, 2.6]))
    assert arm.ik(2.0, 0.0) is None          # 超臂展
    assert arm.ik(0.02, 0.0) is None         # 太近(奇异)


def test_ik_prefers_elbow_down_for_high_targets():
    arm = PlanarArm2R(0.38, 0.40, np.array([-0.3, -2.8]), np.array([2.2, 2.6]))
    q = arm.ik(0.35, 0.30)                   # 高位目标 → 肘下卷分支
    assert q is not None and q[1] < 0


# ============================ 假想底盘硬件 ============================
class MockUnicycleBase:
    """理想差速底盘: v/w 一阶滞后积分, 每控制拍推进 CTRL_DT 秒.

    只为验证 BaseDriver 控制流 —— 不含任何物理仿真.
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self._pose = np.array([x, y, yaw])
        self._vw = (0.0, 0.0)
        self._t = 0.0
        self.ticks = 0

    # RobotInterface 需要的子集
    @property
    def base_pose(self):
        return self._pose.copy()

    def command_drive(self, v, w):
        self._vw = (float(v), float(w))

    def now(self):
        return self._t

    def step_ticks(self, n=1, on_tick=None):
        for _ in range(n):
            self.tick(on_tick)

    def tick(self, on_tick=None):
        v, w = self._vw
        # 一阶滞后: 实际速度用 0.8 倍命令模拟执行器延迟
        x, y, yaw = self._pose
        self._pose = np.array([x + 0.8 * v * np.cos(yaw) * config.CTRL_DT,
                               y + 0.8 * v * np.sin(yaw) * config.CTRL_DT,
                               (yaw + 0.8 * w * config.CTRL_DT + np.pi)
                               % (2 * np.pi) - np.pi])
        self._t += config.CTRL_DT
        self.ticks += 1
        if on_tick:
            on_tick()


def test_base_driver_on_mock_hardware():
    """原语动作: forward(N秒) 按速度×时长推进."""
    hw = MockUnicycleBase(x=0.0, y=0.0, yaw=0.0)
    drv = BaseDriver(hw)
    ok, msg = drv.forward(2.0)                 # 0.30 m/s × 2s × 0.8执行率 = 0.48m
    assert ok, msg
    x, y, yaw = hw.base_pose
    assert abs(x - 0.48) < 0.05
    assert abs(y) < 0.02 and abs(yaw) < 0.05


def test_base_driver_reaches_offset_target():
    """原语动作: turn_left/right 闭环转到位 (编码器反馈, 额外 tick 补偿低速)."""
    hw = MockUnicycleBase(x=0.0, y=0.0, yaw=0.0)
    drv = BaseDriver(hw)
    drv.turn_left(1.0)                          # 目标 1.0 rad, 闭环补偿 0.8 执行率
    _, _, yaw = hw.base_pose
    assert yaw > 0.7                             # 闭环应接近目标(≥0.95 rad, 允许余量)
    yaw_before = yaw
    drv.turn_right(0.5)                         # 目标 -0.5 rad
    _, _, yaw2 = hw.base_pose
    assert abs((yaw2 - yaw_before) + 0.45) < 0.15


@pytest.mark.parametrize("joint,delta", [("shoulder", 0.01), ("elbow", -0.01),
                                        ("shoulder", 0.32), ("elbow", -0.32)])
def test_arm_nudge_applies_exact_goal_including_small_final_remainder(joint, delta):
    initial = np.array([0.2, -0.3])
    commands = []
    hardware = SimpleNamespace(
        arm_q=initial.copy(),
        arm_geometry=lambda: (0.38, 0.40, np.array([-0.3, -2.8]), np.array([2.2, 2.6])),
        now=lambda: 0.0, tick=lambda on_tick=None: None,
        step_ticks=lambda n, on_tick=None: None,
    )

    def command(q):
        commands.append(q.copy())
        hardware.arm_q = q.copy()  # Ideal actuator, so only the commanded goal is under test.

    hardware.command_arm = command
    ok, message = ArmDriver(hardware).nudge(joint, delta)
    expected = initial.copy()
    expected[0 if joint == "shoulder" else 1] += delta
    assert ok, message
    assert commands, "A valid small nudge must still issue a command"
    np.testing.assert_allclose(commands[-1], expected, atol=1e-12)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
