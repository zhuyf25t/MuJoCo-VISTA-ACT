"""MuJoCo 仿真后端: 实现 hal 抽象接口 (MjRobot / MjWorld).

所有 mujoco 读写、物理步进、车头相机渲染与"运动学吸附"封闭在本模块.
自动吸附规则(参考车 try_grasp 的机制): 手指闭合且球距 TCP < GRASP_DIST
→ 吸附; 手指张开 → 释放. 这是物理抓取的仿真等效实现, 对大脑只暴露
"持有/未持有"状态, 不暴露任何球坐标.
"""

from __future__ import annotations

import mujoco
import numpy as np

import config
from hal import RobotInterface, TickCallback, WorldInterface

STEPS_PER_CTRL = int(round(config.CTRL_DT / config.SIM_DT))


class MjRobot(RobotInterface):
    """差速底盘 + 两自由度臂 + 车头相机的 MuJoCo 实现."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        m = model
        self.i_free = m.joint(config.JOINT_BASE).qposadr[0]
        self.i_base = m.body(config.BODY_BASE).id
        self.i_wl = m.actuator(config.ACT_WHEEL_L).id
        self.i_wr = m.actuator(config.ACT_WHEEL_R).id
        self.i_act_arm = np.array([m.actuator(config.ACT_SHOULDER).id,
                                   m.actuator(config.ACT_ELBOW).id])
        self.i_act_fi = m.actuator(config.ACT_FINGER).id
        self.i_tcp = m.site(config.SITE_TCP).id
        self.i_grip_body = m.body(config.BODY_GRIPPER).id
        self.i_arm_base = m.body(config.BODY_ARM_BASE).id
        self.j_sh = m.joint(config.JOINT_SHOULDER).qposadr[0]
        self.j_el = m.joint(config.JOINT_ELBOW).qposadr[0]
        self._ball_ids = {n: m.body(n).id for n in config.BALL_NAMES}
        self._scratch = mujoco.MjData(model)
        self._measure_arm()
        self._attach: tuple[str, np.ndarray] | None = None
        self._front_renderer = None
        self._front_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA,
                                               config.CAM_FRONT)

    def _measure_arm(self) -> None:
        ik = self._scratch
        mujoco.mj_resetData(self.model, ik)
        mujoco.mj_forward(self.model, ik)
        elbow = ik.body(f"{config.ARM_PREFIX}elbow").xpos
        base = ik.body(config.BODY_ARM_BASE).xpos
        tcp = ik.site(config.SITE_TCP).xpos
        self.L1 = float(np.linalg.norm(elbow - base))
        self.L2 = float(np.linalg.norm(tcp - elbow))

    # ================= 状态 =================
    @property
    def base_pose(self) -> np.ndarray:
        q = self.data.qpos
        x, y = q[self.i_free], q[self.i_free + 1]
        qw, qx, qy, qz = q[self.i_free + 3: self.i_free + 7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return np.array([x, y, yaw])

    @property
    def arm_q(self) -> np.ndarray:
        return np.array([self.data.qpos[self.j_sh], self.data.qpos[self.j_el]])

    @property
    def base_velocity(self) -> np.ndarray:
        # Refresh derived velocity at the current qpos/qvel without advancing time.
        mujoco.mj_forward(self.model, self.data)
        velocity = np.empty(6)
        # XBODY uses the base origin/axes, not the body's inertial COM frame.
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                self.i_base, velocity, 1)
        return np.array([velocity[3], velocity[2]])  # rot:lin, local +x/+z

    @property
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.i_tcp].copy()

    @property
    def arm_base_pos(self) -> np.ndarray:
        return self.data.xpos[self.i_arm_base].copy()

    @property
    def finger_opening(self) -> float:
        s = self.data.qpos[self.model.joint(config.JOINT_FINGER).qposadr[0]]
        _, hi = self.finger_ctrl_range()
        return float(np.clip(1.0 - s / hi, 0.0, 1.0)) if hi > 0 else 0.0

    def arm_geometry(self) -> tuple[float, float, np.ndarray, np.ndarray]:
        lo = self.model.actuator_ctrlrange[self.i_act_arm, 0]
        hi = self.model.actuator_ctrlrange[self.i_act_arm, 1]
        return self.L1, self.L2, lo, hi

    # ================= 感知 =================
    def front_image(self) -> np.ndarray:
        """渲染车头相机一帧 (离屏 EGL)."""
        if self._front_renderer is None:
            w, h = config.CAM_FRONT_RES
            self._front_renderer = mujoco.Renderer(self.model, height=h, width=w)
        mujoco.mj_forward(self.model, self.data)     # 确保场景与 qpos 同步
        self._front_renderer.update_scene(self.data, camera=self._front_cam_id)
        return self._front_renderer.render().copy()

    def close(self) -> None:
        if self._front_renderer is not None:
            self._front_renderer.close()
            self._front_renderer = None

    # ================= 低层命令 =================
    def command_drive(self, v: float, w: float) -> None:
        wl = (v - w * 0.42 / 2) / 0.1
        wr = (v + w * 0.42 / 2) / 0.1
        self.data.ctrl[self.i_wl] = float(np.clip(wl, -5, 5))
        self.data.ctrl[self.i_wr] = float(np.clip(wr, -5, 5))

    def command_arm(self, q: np.ndarray) -> None:
        self.data.ctrl[self.i_act_arm] = np.asarray(q, dtype=float)

    def command_finger(self, goal_ctrl: float) -> None:
        self.data.ctrl[self.i_act_fi] = float(goal_ctrl)

    def finger_ctrl_range(self) -> tuple[float, float]:
        rng = self.model.actuator_ctrlrange[self.i_act_fi]
        return float(rng[0]), float(rng[1])

    @property
    def applied_action(self) -> np.ndarray:
        # MuJoCo may clamp actuator inputs even when data.ctrl is out of range.
        ctrl = self.data.ctrl.copy()
        limited = self.model.actuator_ctrllimited.astype(bool)
        ctrl[limited] = np.clip(ctrl[limited], self.model.actuator_ctrlrange[limited, 0],
                                self.model.actuator_ctrlrange[limited, 1])
        lo, hi = self.finger_ctrl_range()
        left, right = ctrl[self.i_wl], ctrl[self.i_wr]
        return np.array([*ctrl[self.i_act_arm], (hi - ctrl[self.i_act_fi]) / (hi - lo),
                         0.1 / 2 * (left + right), 0.1 / 0.42 * (right - left)])

    # ================= 时间 =================
    def now(self) -> float:
        return float(self.data.time)

    def tick(self, on_tick: TickCallback | None = None) -> None:
        if self.before_tick is not None:
            self.before_tick()
        for _ in range(STEPS_PER_CTRL):
            mujoco.mj_step(self.model, self.data)
            self._auto_grasp()
            self._apply_attach()
        if on_tick:
            on_tick()

    # ================= 自动吸附 (物理抓取的仿真等效) =================
    def _finger_ball_contacts(self, ball_bid: int) -> int:
        """手指 pad 与指定球之间的接触数."""
        n = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            hit = False
            if "finger" in g1 and self.model.geom_bodyid[c.geom2] == ball_bid:
                hit = True
            elif "finger" in g2 and self.model.geom_bodyid[c.geom1] == ball_bid:
                hit = True
            if hit:
                n += 1
        return n

    def _auto_grasp(self) -> None:
        """手指闭合 + 球在夹爪附近 + **pad 物理接触** → 持有.
        接触检测是必须的: 球必须在两指之间被实际夹住, 而非隔空吸附."""
        closed = self.data.ctrl[self.i_act_fi] >= config.FINGER_CLOSED_THRESHOLD
        if self._attach is not None:
            if not closed:
                self._attach = None
            return
        if not closed:
            return
        tcp = self.data.site_xpos[self.i_tcp]
        for name, bid in self._ball_ids.items():
            if np.linalg.norm(self.data.xpos[bid] - tcp) > config.GRASP_DIST:
                continue
            # 必须有手指 pad 与球的物理接触(≥1 个接触点)
            if self._finger_ball_contacts(bid) >= 1:
                self.attach_object(name)
                return

    def attach_object(self, name: str) -> None:
        g = self.i_grip_body
        gmat = np.array(self.data.xmat[g]).reshape(3, 3)
        relpos = gmat.T @ (self.data.xpos[self.model.body(name).id] - self.data.xpos[g])
        self._attach = (name, relpos)

    def detach_object(self) -> None:
        self._attach = None

    def attached_object(self) -> str | None:
        return self._attach[0] if self._attach is not None else None

    def _apply_attach(self) -> None:
        if self._attach is None:
            return
        name, relpos = self._attach
        g = self.i_grip_body
        gmat = np.array(self.data.xmat[g]).reshape(3, 3)
        bpos = self.data.xpos[g] + gmat @ relpos
        j = self.model.joint(f"{name}_joint")
        qadr, dadr = j.qposadr[0], j.dofadr[0]
        self.data.qpos[qadr:qadr + 3] = bpos
        self.data.qvel[dadr:dadr + 6] = 0.0


class MjWorld(WorldInterface):
    """任务评估 (真值, 仅回合判定用)."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data

    def object_in_bin(self, name: str) -> bool:
        pos = self.data.xpos[self.model.body(name).id]
        vadr = self.model.joint(f"{name}_joint").dofadr[0]
        vlin = self.data.qvel[vadr:vadr + 3]
        bin_x, bin_y = config.BIN_POS
        hx, hy = config.BIN_HALF_INT
        inside_xy = abs(pos[0] - bin_x) < hx and abs(pos[1] - bin_y) < hy
        inside_z = 0.012 < pos[2] < config.BIN_WALL_TOP
        slow = np.linalg.norm(vlin) < 0.08
        return bool(inside_xy and inside_z and slow)
