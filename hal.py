"""硬件抽象层 (HAL): 控制逻辑与仿真/真机之间的边界.

control/ 包只依赖本文件定义的接口 —— 不 import mujoco.
- RobotInterface: 低层命令(轮速/臂舵机/手指) + 状态读取 + 车头相机 + 时间
- WorldInterface: 任务判定 (真值, 仅用于评估, 大脑不可见)

感知约定: 大脑接收车头图像与白名单本体状态, 没有任何物体真值坐标.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

TickCallback = Callable[[], None]


class RobotInterface(ABC):
    """机器人硬件接口: 低层命令 + 状态 + 感知 + 时间."""

    # ---------- 状态(本体感觉) ----------
    @property
    @abstractmethod
    def base_pose(self) -> np.ndarray:
        """底盘位姿 (x, y, yaw) —— 里程计."""

    @property
    @abstractmethod
    def base_velocity(self) -> np.ndarray:
        """实测 [v, w] (m/s, rad/s)，底盘原点、车体坐标系 +x 前进/+z 左转。

        真机由传感器/里程计提供；不能返回 command_drive 的目标命令。
        """

    @property
    @abstractmethod
    def arm_q(self) -> np.ndarray:
        """臂关节角 [肩, 肘] (rad) —— 编码器."""

    @property
    @abstractmethod
    def tcp_pos(self) -> np.ndarray:
        """夹爪 TCP 世界坐标 —— 正运动学."""

    @property
    @abstractmethod
    def arm_base_pos(self) -> np.ndarray:
        """肩轴世界坐标."""

    @property
    @abstractmethod
    def finger_opening(self) -> float:
        """手指开度 0=闭合 1=全开."""

    @abstractmethod
    def arm_geometry(self) -> tuple[float, float, np.ndarray, np.ndarray]:
        """臂几何 (L1, L2, q_lo, q_hi)."""

    # ---------- 感知 ----------
    @abstractmethod
    def front_image(self) -> np.ndarray:
        """车头前向相机一帧 RGB (H,W,3) uint8 —— 机器人唯一的"眼睛"."""

    # ---------- 低层命令 ----------
    @abstractmethod
    def command_drive(self, v: float, w: float) -> None:
        """底盘线速度/角速度开环命令 (m/s, rad/s)."""

    @abstractmethod
    def command_arm(self, q: np.ndarray) -> None:
        """臂舵机目标角 [肩, 肘] (rad)."""

    @abstractmethod
    def command_finger(self, goal_ctrl: float) -> None:
        """手指滑动目标 (后端自定义标度)."""

    @abstractmethod
    def finger_ctrl_range(self) -> tuple[float, float]:
        """手指命令取值范围."""

    # ---------- 时间 ----------
    @abstractmethod
    def now(self) -> float:
        """当前时刻 (s). 仿真=物理时间; 真机=墙钟."""

    @abstractmethod
    def tick(self, on_tick: TickCallback | None = None) -> None:
        """推进一个控制拍 (10Hz)."""

    def step_ticks(self, n: int = 1, on_tick: TickCallback | None = None) -> None:
        for _ in range(n):
            self.tick(on_tick)

    # ---------- 夹持感知 ----------
    # 指闭合且物体在夹爪内 → 后端登记"持有"; 张开 → 释放.
    # 仿真: 闭合+球在 TCP 0.22m 内 = 运动学吸附(物理抓取的等效实现);
    # 真机: 力/电流传感器判断, 吸附/解除只是登记状态.
    @abstractmethod
    def attach_object(self, name: str) -> None:
        """按当前夹爪-物体相对位姿吸附物体."""

    @abstractmethod
    def detach_object(self) -> None:
        """解除吸附."""

    @abstractmethod
    def attached_object(self) -> str | None:
        """当前持有的物体名(仿真内部标识)或 None."""


class WorldInterface(ABC):
    """任务评估接口: 真值判定 —— 只在回合结束时用, 大脑不可见."""

    @abstractmethod
    def object_in_bin(self, name: str) -> bool:
        """物体是否已落入收纳箱."""
