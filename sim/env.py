"""仿真环境: 组装 MuJoCo 后端 + 控制驱动, 提供兼容门面.

感知约定: 大脑接收车头相机与白名单本体状态; env.success() 用真值,
仅在回合结束时判定, 不进入任何决策路径.
"""

from __future__ import annotations

import mujoco
import numpy as np

import config
from control.drivers import ArmDriver, BaseDriver
from hal import WorldInterface
from .backend import MjRobot, MjWorld


class TaskSpec:
    """捡球任务: 把一个网球捡起来放进收纳箱 (视觉无法区分球编号)."""

    def text(self) -> str:
        return "用视觉找到场上的网球, 捡起来放进洋红色收纳箱"


def parse_task(text: str | None = None) -> TaskSpec:
    return TaskSpec()


class SimRobotFacade:
    """组合 [MjRobot 后端 + 控制驱动], 对外提供上层所需的状态与动作转发."""

    def __init__(self, hal: MjRobot, world: WorldInterface):
        self.hal = hal
        self.world = world
        self.base = BaseDriver(hal)
        self.arm = ArmDriver(hal, world)

    # ---- 状态直通 ----
    @property
    def base_pose(self) -> np.ndarray:
        return self.hal.base_pose

    @property
    def base_velocity(self) -> np.ndarray:
        return self.hal.base_velocity

    @property
    def arm_q(self) -> np.ndarray:
        return self.hal.arm_q

    @property
    def tcp_pos(self) -> np.ndarray:
        return self.hal.tcp_pos

    @property
    def finger_opening(self) -> float:
        return self.hal.finger_opening

    def welded_ball(self):                      # 兼容旧名 = 持有状态
        return self.hal.attached_object()

    def step_ctrl(self, n=1, on_tick=None):
        self.hal.step_ticks(n, on_tick)

    def front_image(self) -> np.ndarray:
        return self.hal.front_image()


class MobileManipEnv:
    """移动捡球机器人 + 地面场景 (MuJoCo 后端, 车头相机感知)."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(config.SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.hal = MjRobot(self.model, self.data)
        self.world: WorldInterface = MjWorld(self.model, self.data)
        self.robot = SimRobotFacade(self.hal, self.world)
        self.task = TaskSpec()
        self.rng = np.random.default_rng()

    # ---------- 生命周期 ----------
    def close(self) -> None:
        """在图形系统退出之前释放车头相机; 可重复调用."""
        self.hal.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def reset(self, seed: int | None = None, task: TaskSpec | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if task is not None:
            self.task = task
        m, d, r = self.model, self.data, self.robot
        mujoco.mj_resetData(m, d)
        r.hal.detach_object()

        bx, by, byaw = config.BASE_SPAWN
        d.qpos[r.hal.i_free] = bx + self.rng.normal(0, 0.03)
        d.qpos[r.hal.i_free + 1] = by + self.rng.normal(0, 0.03)
        d.qpos[r.hal.i_free + 2] = 0.1
        cy, sy = np.cos(byaw / 2), np.sin(byaw / 2)
        d.qpos[r.hal.i_free + 3:r.hal.i_free + 7] = [cy, 0, 0, sy]

        d.qpos[[r.hal.j_sh, r.hal.j_el]] = config.ARM_STOW
        d.ctrl[r.hal.i_act_arm] = config.ARM_STOW
        d.ctrl[r.hal.i_act_fi] = config.FINGER_OPEN_CTRL
        self._randomize_balls()
        mujoco.mj_forward(m, d)
        for _ in range(500):
            mujoco.mj_step(m, d)

    def _randomize_balls(self) -> None:
        bin_x, bin_y = config.BIN_POS
        placed: list[tuple[float, float]] = []
        for name in config.BALL_NAMES:
            pos = None
            for _ in range(500):
                x = self.rng.uniform(*config.BALL_SPAWN_X)
                y = self.rng.uniform(*config.BALL_SPAWN_Y)
                if np.hypot(x - bin_x, y - bin_y) < 0.45:
                    continue
                if x < config.BASE_SPAWN[0] + 0.6 and abs(y) < 0.4:
                    continue
                if all(np.hypot(x - px, y - py) > 0.25 for px, py in placed):
                    pos = (x, y)
                    break
            if pos is None:
                pos = (0.8 + 0.25 * len(placed), -0.6 + 0.35 * len(placed))
            placed.append(pos)
            qadr = self.model.joint(f"{name}_joint").qposadr[0]
            self.data.qpos[qadr] = pos[0]
            self.data.qpos[qadr + 1] = pos[1]
            self.data.qpos[qadr + 2] = config.BALL_R + 0.004
            self.data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
        for name in config.BALL_NAMES:
            gid = self.model.geom(f"{name}_geom").id
            k = 1.0 + self.rng.uniform(-config.RAND_PHYS, config.RAND_PHYS)
            self.model.body_mass[self.model.body(name).id] = 0.06 * k
            self.model.geom_friction[gid, 0] = 0.9 * k

    # ---------- 观测 / 评估 ----------
    def get_obs(self, images: dict | None = None) -> dict:
        """同一仿真时刻的图像和状态；调用期间不推进物理。

        若传入 images，调用方须保证它来自当前时刻。
        """
        r = self.robot
        images = images if images is not None else {"front": self.hal.front_image()}
        v, w = r.base_velocity
        shoulder, elbow = r.arm_q
        opening = r.finger_opening
        return {
            "images": images,
            "timestamp_s": self.hal.now(),
            "observation": {
                "shoulder_rad": float(shoulder), "elbow_rad": float(elbow),
                "gripper_opening": opening,
                "base_v_mps": float(v), "base_w_radps": float(w),
            },
            "arm_qpos": [float(shoulder), float(elbow)],
            "finger": opening,
            "base_pose": r.base_pose.tolist(),
            "tcp_pos": r.tcp_pos.tolist(),
        }

    def success(self) -> bool:
        """真值评估: 任一球入箱即成功 (仅回合判定, 大脑不可见)."""
        return any(self.world.object_in_bin(n) for n in config.BALL_NAMES)
