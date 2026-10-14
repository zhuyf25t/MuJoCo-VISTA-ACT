"""历史轨迹读取与状态恢复，供 GUI 回放和视频重渲染共用。"""

from __future__ import annotations

import json
from itertools import chain
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass
class Frame:
    t: float
    qpos: np.ndarray
    ctrl: np.ndarray


def trajectory_path(source: str, root: Path) -> Path:
    """接受 episode 名、episode 目录或 JSONL 文件路径。"""
    path = Path(source).expanduser()
    if not path.exists() and len(path.parts) == 1 and path.suffix != ".jsonl":
        path = root / path
    return path if path.suffix == ".jsonl" else path / "trajectory.jsonl"


def load_trajectory(path: Path, model: mujoco.MjModel) -> list[Frame]:
    frames = []
    with path.open(encoding="utf-8-sig") as stream:
        # New records store pre-action frames; retain the final post-action pose.
        terminal = path.with_name("terminal.json")
        tail = [terminal.read_text(encoding="utf-8")] if path.name == "trajectory.jsonl" and terminal.is_file() else []
        for line_number, line in enumerate(chain(stream, tail), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                t = float(row["t"])
                qpos = np.asarray(row["qpos"], dtype=float)
                ctrl = np.asarray(row.get("ctrl", np.zeros(model.nu)), dtype=float)
                if qpos.shape != (model.nq,):
                    raise ValueError(f"qpos 维度应为 {model.nq}，实际为 {qpos.shape}；请使用录制时的模型")
                if ctrl.shape != (model.nu,):
                    raise ValueError(f"ctrl 维度应为 {model.nu}，实际为 {ctrl.shape}")
                if not (np.isfinite(t) and np.isfinite(qpos).all() and np.isfinite(ctrl).all()):
                    raise ValueError("时间或状态含非有限数值")
                if t < 0 or (frames and t < frames[-1].t):
                    raise ValueError("时间必须非负且按先后顺序排列")
            except (ValueError, TypeError, KeyError) as e:
                raise ValueError(f"{path} 第 {line_number} 行: {e}") from e
            frames.append(Frame(t, qpos, ctrl))
    if not frames:
        raise ValueError(f"轨迹为空: {path}")
    return frames


def apply_frame(model: mujoco.MjModel, data: mujoco.MjData, frame: Frame) -> None:
    """恢复记录中的位置，不调用 mj_step，也不重新执行 brain/工具。"""
    data.qpos[:] = frame.qpos
    data.qvel[:] = 0  # 旧记录没有速度；回放仅重建姿态，不续跑物理。
    data.ctrl[:] = frame.ctrl
    data.time = frame.t
    mujoco.mj_forward(model, data)


class PlaybackCursor:
    """按录制时间推进；GUI 暂停与重播不依赖渲染帧率。"""

    def __init__(self, frames: list[Frame], speed: float = 1.0, loop: bool = False):
        if not frames:
            raise ValueError("轨迹为空")
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("倍速必须是大于 0 的有限数值")
        self.times = [frame.t for frame in frames]
        self.speed = speed
        self.loop = loop
        self.restart()

    @property
    def index(self) -> int:
        return max(0, bisect_right(self.times, self.position) - 1)

    def restart(self) -> None:
        self.position = self.times[0]
        self.paused = False
        self.finished = False

    def toggle_pause(self) -> None:
        if self.finished:
            self.restart()
        else:
            self.paused = not self.paused

    def advance(self, elapsed: float) -> None:
        if self.paused:
            return
        self.position += max(0.0, elapsed) * self.speed
        duration = self.times[-1] - self.times[0]
        if self.position >= self.times[-1]:
            if self.loop and duration > 0:
                self.position = self.times[0] + (self.position - self.times[0]) % duration
            else:
                self.position = self.times[-1]
                self.paused = self.finished = True
