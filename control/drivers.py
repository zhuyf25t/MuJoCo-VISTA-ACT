"""控制逻辑: 时间盒动作原语 + 臂预设 + 手指.

只依赖 hal 抽象接口与纯数学 —— 不 import mujoco.
动作风格: forward/back/turn_left/turn_right(秒) 等低层原语,
没有 move-to 式的坐标目标; 闭环由大脑看图决策完成.
"""

from __future__ import annotations

import numpy as np

import config
from hal import RobotInterface, TickCallback
from .kinematics import PlanarArm2R


class ActionFailed(Exception):
    """A commanded motion finished its attempt but did not reach its target.

    All executed ticks completed normally. The command was applied; this is
    recoverable from a fresh observation, unlike an exception during a tick.
    """


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ============================ 底盘 ============================
class BaseDriver:
    """时间盒动作原语: 每次调用按固定速度执行 N 秒."""

    def __init__(self, robot: RobotInterface):
        self.r = robot

    def _run(self, seconds: float, v: float, w: float,
             on_tick: TickCallback | None = None) -> tuple[bool, str]:
        """前进/后退=纯时间盒; 转向=里程计闭环(比例减速+到位即停, 防过冲振荡)."""
        seconds = float(np.clip(seconds, 0.05, config.PRIM_MAX_S))
        n_total = max(1, round(seconds / config.CTRL_DT))
        yaw0 = self.r.base_pose[2]
        if w == 0.0:
            # 前进/后退: 简单时间盒
            self.r.command_drive(v, w)
            self.r.step_ticks(n_total, on_tick)
        else:
            # 转向: 编码器闭环
            target = abs(w) * seconds
            for _ in range(n_total + 15):
                actual = abs(wrap_angle(self.r.base_pose[2] - yaw0))
                if actual >= target * 0.95:
                    break
                progress = actual / max(target, 1e-6)
                rate = 1.0 if progress < 0.6 else max(0.3, 1.5 - progress)
                self.r.command_drive(v, w * rate)
                self.r.tick(on_tick)
        self.r.command_drive(0.0, 0.0)
        self.r.step_ticks(2, on_tick)
        act = ("前进" if v > 0 else "后退") if v else ("左转" if w > 0 else "右转")
        extra = ""
        if w != 0.0:
            actual_deg = np.degrees(wrap_angle(self.r.base_pose[2] - yaw0))
            extra = f" (实际转{actual_deg:+.0f}°)"
        return True, f"{act} {seconds:.1f}s{extra} 完成, 当前位姿 {np.round(self.r.base_pose, 2).tolist()}"

    def forward(self, seconds: float = 1.0,
                on_tick: TickCallback | None = None) -> tuple[bool, str]:
        return self._run(seconds, config.PRIM_V, 0.0, on_tick)

    def back(self, seconds: float = 1.0,
             on_tick: TickCallback | None = None) -> tuple[bool, str]:
        return self._run(seconds, -config.PRIM_V, 0.0, on_tick)

    def turn_left(self, seconds: float = 1.0,
                  on_tick: TickCallback | None = None) -> tuple[bool, str]:
        return self._run(seconds, 0.0, config.PRIM_W, on_tick)

    def turn_right(self, seconds: float = 1.0,
                   on_tick: TickCallback | None = None) -> tuple[bool, str]:
        return self._run(seconds, 0.0, -config.PRIM_W, on_tick)


# ============================ 臂 ============================
class ArmDriver:
    """两自由度臂: 预设姿态 + 关节微调 + 手指开合."""

    def __init__(self, robot: RobotInterface, world=None):
        self.r = robot
        l1, l2, lo, hi = robot.arm_geometry()
        self.kin = PlanarArm2R(l1, l2, lo, hi)

    # ---------- 预设姿态 ----------
    _PRESETS = {"stow": config.ARM_STOW, "carry": config.ARM_CARRY,
                "reach": config.ARM_REACH, "drop": config.ARM_DROP}

    def move_arm_q(self, q_target, on_tick: TickCallback | None = None,
                   timeout: float = 6.0) -> tuple[bool, str]:
        q_goal = np.clip(np.asarray(q_target, dtype=float),
                         self.kin.q_lo, self.kin.q_hi)
        q_cmd = self.r.arm_q.copy()
        rate = 0.3
        t0 = self.r.now()
        while np.linalg.norm(q_cmd - q_goal) > 0.03:
            q_cmd = np.clip(q_goal, q_cmd - rate, q_cmd + rate)
            self.r.command_arm(q_cmd)
            if self.r.now() - t0 > timeout:
                raise ActionFailed(f"臂移动超时 (目标 {np.round(q_goal, 2)})")
            self.r.tick(on_tick)
        # The ramp tolerance must not discard a small nudge or its final remainder.
        self.r.command_arm(q_goal)
        self.r.step_ticks(3, on_tick)
        err = float(np.linalg.norm(self.r.arm_q - q_goal))
        if err > 0.35:
            raise ActionFailed(f"臂被卡住 (残差 {np.degrees(err):.0f}°)")
        return True, f"臂到位 (残差 {np.degrees(err):.0f}°)"

    def move_pose(self, pose: str, on_tick: TickCallback | None = None) -> tuple[bool, str]:
        if pose not in self._PRESETS:
            return False, f"未知姿态 {pose} (可选: {'/'.join(self._PRESETS)})"
        ok, msg = self.move_arm_q(self._PRESETS[pose], on_tick=on_tick)
        return ok, f"臂到 {pose} 姿态: {msg}"

    # ---------- 关节微调 ----------
    def nudge(self, joint: str, delta: float,
              on_tick: TickCallback | None = None) -> tuple[bool, str]:
        """joint: 'shoulder'|'elbow'; delta 为关节角增量（弧度），不表示夹爪升降。"""
        delta = float(np.clip(delta, -config.ARM_NUDGE_MAX, config.ARM_NUDGE_MAX))
        q = self.r.arm_q.copy()
        i = 0 if joint == "shoulder" else 1
        q[i] += delta
        ok, msg = self.move_arm_q(q, on_tick=on_tick, timeout=3.0)
        return ok, f"{joint} 角度增量指令 {delta:+.2f} rad: {msg}"

    # ---------- 手指 ----------
    def open_gripper(self, on_tick: TickCallback | None = None) -> tuple[bool, str]:
        return self._finger(config.FINGER_OPEN_CTRL, on_tick)

    def close_gripper(self, on_tick: TickCallback | None = None) -> tuple[bool, str]:
        return self._finger(config.FINGER_CLOSE_CTRL, on_tick)

    def _finger(self, goal: float, on_tick: TickCallback | None) -> tuple[bool, str]:
        lo, hi = self.r.finger_ctrl_range()
        goal = float(np.clip(goal, lo, hi))
        self.r.command_finger(goal)
        self.r.step_ticks(8, on_tick)                # 0.8 s 行程 + 稳定
        state = "闭合" if goal >= config.FINGER_CLOSED_THRESHOLD else "张开"
        held = self.r.attached_object() is not None
        if goal >= config.FINGER_CLOSED_THRESHOLD:
            # 闭合: 明确反馈抓取成败, 模型据此决定重试或继续
            if held:
                return True, "手指已闭合, ✓ 成功夹到球!"
            return True, ("手指已闭合, ✗ 没有夹到球. "
                          "按结果里的'夹爪视角'小步修正(0.2~0.3s)后重试抓取序列")
        return True, f"手指已{state}" + (", 球已释放" if not held else "")
