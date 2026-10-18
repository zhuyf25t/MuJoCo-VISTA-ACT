"""仿真环境: 组装 MuJoCo 后端 + 控制驱动, 提供兼容门面.

感知约定: 大脑接收车头相机与白名单本体状态; env.success() 用真值,
仅在回合结束时判定, 不进入任何决策路径.
"""

from __future__ import annotations

import mujoco
import numpy as np

import config
from observations import FrameObservation, ObservationWindow, OBSERVATION_FPS
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
        self._observation_stream_active = False
        self._observation_buffer: list[FrameObservation] = []
        self._observation_time: float | None = None
        self._frame_no = 0

    # ---------- 生命周期 ----------
    def close(self) -> None:
        """在图形系统退出之前释放车头相机; 可重复调用."""
        self.hal.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def reset(self, seed: int | None = None, task: TaskSpec | None = None) -> None:
        self._observation_stream_active = False
        self._observation_buffer = []
        self._observation_time = None
        self._frame_no = 0
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
    def _capture_frame(self, frame_no: int) -> FrameObservation:
        """Capture public measurements and RGB at the same stopped physics boundary."""
        r = self.robot
        v, w = r.base_velocity
        shoulder, elbow = r.arm_q
        return FrameObservation(frame_no, self.hal.front_image(),
                                (float(shoulder), float(elbow), r.finger_opening, float(v), float(w)))

    @property
    def observation_stream_active(self) -> bool:
        return self._observation_stream_active

    @property
    def current_frame_no(self) -> int:
        return self._frame_no

    def start_observation_stream(self) -> None:
        if self._observation_stream_active or self._observation_buffer:
            raise RuntimeError("VISTA 观测流已经开始；新 episode 必须先 reset")
        if not np.isclose(config.CTRL_DT, 1 / OBSERVATION_FPS, rtol=0, atol=1e-12):
            raise RuntimeError("VISTA 帧格式要求控制频率为 10Hz")
        initial = self._capture_frame(0)
        self._observation_buffer = [initial]
        self._frame_no = 0
        self._observation_time = self.hal.now()
        self._observation_stream_active = True

    def capture_observation_tick(self) -> None:
        if not self._observation_stream_active:
            raise RuntimeError("VISTA 观测流未开始")
        if self._observation_buffer[-1].frame_no != self._frame_no:
            raise RuntimeError("OBSERVATION_UNAVAILABLE: 上一个控制拍采样失败，不能跳过缺帧继续采集")
        now = self.hal.now()
        if not np.isclose(now - self._observation_time, config.CTRL_DT, rtol=0, atol=1e-8):
            raise RuntimeError("OBSERVATION_UNAVAILABLE: 采样必须恰好相隔一个完整控制拍")
        # Physics has already completed this tick, even if rendering now fails.
        self._frame_no += 1
        self._observation_time = now
        frame = self._capture_frame(self._frame_no)
        self._observation_buffer.append(frame)

    def get_obs(self) -> ObservationWindow:
        """Read the unacknowledged inclusive window without advancing or consuming it."""
        if not self._observation_buffer:
            raise RuntimeError("VISTA 观测流尚无初始帧")
        return ObservationWindow(tuple(self._observation_buffer))

    def ack_observations(self, frame_no: int) -> None:
        start, end = self._observation_buffer[0].frame_no, self._observation_buffer[-1].frame_no
        if type(frame_no) is not int or not start <= frame_no <= end:
            raise RuntimeError("VISTA 观测确认超出缓存范围")
        self._observation_buffer = self._observation_buffer[frame_no - start:]

    def stop_observation_stream(self) -> None:
        self._observation_stream_active = False

    def success(self) -> bool:
        """真值评估: 任一球入箱即成功 (仅回合判定, 大脑不可见)."""
        return any(self.world.object_in_bin(n) for n in config.BALL_NAMES)
