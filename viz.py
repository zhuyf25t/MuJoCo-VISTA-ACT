"""渲染封装: EGL 离屏相机组 + GUI viewer 包装."""

from __future__ import annotations

from contextlib import ExitStack

import numpy as np
import mujoco

import config


class CameraRig:
    """每个观测相机一个离屏 Renderer (EGL)."""

    def __init__(self, model: mujoco.MjModel,
                 cams: list[str] | None = None,
                 width: int = 640, height: int = 480):
        self.cams = cams or config.OBS_CAMS
        self._renderers = {}
        self._cam_ids = {}
        with ExitStack() as resources:
            for cam in self.cams:
                cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
                if cid < 0:
                    raise ValueError(f"相机不存在: {cam}")
                renderer = mujoco.Renderer(model, height=height, width=width)
                resources.callback(renderer.close)
                self._renderers[cam] = renderer
                self._cam_ids[cam] = cid
            self._resources = resources.pop_all()

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        out = {}
        for c, r in self._renderers.items():
            r.update_scene(data, camera=self._cam_ids[c])
            out[c] = r.render()
        return out

    def close(self) -> None:
        self._resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class GuiViewer:
    """launch_passive 包装: 主循环跑物理, 每控制拍 sync()."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        import mujoco.viewer
        self._viewer = mujoco.viewer.launch_passive(model, data)
        v = self._viewer
        v.cam.lookat[:] = [0.55, 0.0, 0.55]
        v.cam.distance = 2.2
        v.cam.azimuth = 130.0
        v.cam.elevation = -25.0

    def sync(self) -> None:
        if self._viewer.is_running():
            self._viewer.sync()

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def set_status(self, text: str) -> None:
        """显示采集状态；旧版 MuJoCo 没有此接口时跳过。"""
        if hasattr(self._viewer, "set_texts"):
            self._viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, text, ""))

    def close(self) -> None:
        try:
            self._viewer.close()
        except Exception:
            pass

    def __enter__(self) -> "GuiViewer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
