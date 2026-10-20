"""MuJoCo 仿真后端: 实现 hal 抽象接口 (MjRobot / MjWorld).

所有 mujoco 读写、物理步进、车头相机渲染与"运动学吸附"封闭在本模块.
闭合目标持续生效时，实测开度稳定且同一球持续接触两侧指垫内表面后，才启用
辅助吸附；开爪或持续失去有效夹持时释放。候选及吸附状态留在后端，
VISTA 仍通过图像和实测关节开度判断持球。
"""

from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

import config
from hal import RobotInterface, TickCallback, WorldInterface

STEPS_PER_CTRL = int(round(config.CTRL_DT / config.SIM_DT))
FRONT_RENDER_PROTOCOL = "front_renderer_one_frame_warmup_v1"


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
        self._ball_geoms = {m.geom(f"{n}_geom").id: n for n in config.BALL_NAMES}
        # 指垫局部 y 轴上的内表面方向：活动左指 -y，固定右指 +y。
        self._finger_geoms = {m.geom(f"{config.ARM_PREFIX}left_finger_geom").id: -1,
                              m.geom(f"{config.ARM_PREFIX}right_finger_geom").id: 1}
        self._scratch = mujoco.MjData(model)
        self._measure_arm()
        self._attach: tuple[str, np.ndarray] | None = None
        self._grasp_candidate: str | None = None
        self._grasp_samples: deque[tuple[float, float]] = deque()
        self._grasp_lost_since: float | None = None
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
        created = self._front_renderer is None
        if created:
            w, h = config.CAM_FRONT_RES
            self._front_renderer = mujoco.Renderer(self.model, height=h, width=w)
        mujoco.mj_forward(self.model, self.data)     # 确保场景与 qpos 同步
        if created:
            # EGL's first draw can differ by one pixel from its steady rendering.
            # Warm only the graphics resources: do not advance or refresh physics.
            self._front_renderer.update_scene(self.data, camera=self._front_cam_id)
            self._front_renderer.render()
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
        if goal_ctrl < config.FINGER_CLOSED_THRESHOLD:
            self.detach_object()  # 开爪立即解除，不等下一次接触检测。

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

    # ================= 确认夹持后的辅助吸附 =================
    def _is_inner_pad_contact(self, contact, finger: int) -> bool:
        """接触须位于指垫内表面；排除外侧、指尖及背面碰撞。"""
        rotation = self.data.geom_xmat[finger].reshape(3, 3)
        point = rotation.T @ (contact.pos - self.data.geom_xpos[finger])
        normal = rotation.T @ contact.frame[:3]
        half = self.model.geom_size[finger]
        tol = config.GRASP_CONTACT_TOL_M
        return bool(abs(point[1] - self._finger_geoms[finger] * half[1]) <= tol
                    and abs(point[0]) <= half[0] + tol
                    and abs(point[2]) <= half[2] + tol
                    and abs(normal[1]) >= config.GRASP_NORMAL_COS_MIN)

    def _valid_grasp_balls(self) -> set[str]:
        opening = self.finger_opening
        if not config.GRASP_OPENING_MIN < opening < config.GRASP_OPENING_MAX:
            return set()
        sides: dict[str, set[int]] = {}
        for contact in self.data.contact:
            g1, g2 = contact.geom1, contact.geom2
            if g1 in self._finger_geoms and g2 in self._ball_geoms:
                finger, ball = g1, g2
            elif g2 in self._finger_geoms and g1 in self._ball_geoms:
                finger, ball = g2, g1
            else:
                continue
            if contact.efc_address < 0 or not self._is_inner_pad_contact(contact, finger):
                continue
            # 用左右指垫的 ID 去重：同侧多个接触点不能伪装成双侧接触。
            sides.setdefault(self._ball_geoms[ball], set()).add(finger)
        tcp = self.data.site_xpos[self.i_tcp]
        return {name for name, fingers in sides.items()
                if len(fingers) == 2
                and np.linalg.norm(self.data.xpos[self._ball_ids[name]] - tcp) <= config.GRASP_DIST}

    def _clear_grasp_candidate(self) -> None:
        self._grasp_candidate = None
        self._grasp_samples.clear()

    def _auto_grasp(self) -> None:
        """闭合目标允许持续接触判定；动作等待超时不改变物理目标。"""
        if self.data.ctrl[self.i_act_fi] < config.FINGER_CLOSED_THRESHOLD:
            self.detach_object()
            return
        # mj_step 已推进 qpos；刷新接触和几何，避免与上一步的派生量混用。
        mujoco.mj_forward(self.model, self.data)
        if self._attach is not None:
            # 已确认的辅助约束先跟上当前臂姿态，再复核接触；否则会把球
            # 尚在上一物理步的位置造成的短暂间隙误判为运输中失去夹持。
            self._apply_attach()
            mujoco.mj_forward(self.model, self.data)
        valid = self._valid_grasp_balls()
        now = self.now()
        if self._attach is not None:
            if self._attach[0] in valid:
                self._grasp_lost_since = None
            elif self._grasp_lost_since is None:
                self._grasp_lost_since = now
            elif now - self._grasp_lost_since >= config.GRASP_LOSS_SECONDS - 1e-9:
                self.detach_object()
            return
        candidate = next((name for name in self._ball_ids if name in valid), None)
        if candidate is None:
            self._clear_grasp_candidate()
            return
        if candidate != self._grasp_candidate:
            self._clear_grasp_candidate()
            self._grasp_candidate = candidate
        self._grasp_samples.append((now, self.finger_opening))
        cutoff = now - config.GRASP_STABLE_SECONDS
        while len(self._grasp_samples) > 1 and self._grasp_samples[1][0] <= cutoff:
            self._grasp_samples.popleft()
        elapsed = now - self._grasp_samples[0][0]
        openings = [sample[1] for sample in self._grasp_samples]
        if (elapsed >= config.GRASP_STABLE_SECONDS - 1e-9
                and max(openings) - min(openings) <= config.GRASP_OPENING_TOL):
            self.attach_object(candidate)

    def attach_object(self, name: str) -> None:
        g = self.i_grip_body
        gmat = np.array(self.data.xmat[g]).reshape(3, 3)
        relpos = gmat.T @ (self.data.xpos[self.model.body(name).id] - self.data.xpos[g])
        self._attach = (name, relpos)
        self._grasp_lost_since = None
        self._clear_grasp_candidate()

    def detach_object(self) -> None:
        self._attach = None
        self._grasp_lost_since = None
        self._clear_grasp_candidate()

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
