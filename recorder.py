"""双层数据记录器.

决策级 decisions.jsonl —— LLM 的 thought/tool/args/result 全量;
控制级 trajectory.jsonl + imgs/ —— 固定 CTRL_HZ 频率的 (图像, 状态, 底层动作),
可直接映射为模仿学习的 (observation, action) 对.

当前记录包含两关节臂、夹爪、底盘和完整仿真状态；训练数据集导出尚未实现。
"""

from __future__ import annotations

import io
import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

import config


def _jpg_bytes(rgb: np.ndarray, quality: int = 80) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class EpisodeRecorder:
    """一个 episode 一个目录: meta.json / decisions.jsonl / trajectory.jsonl / imgs/."""

    def __init__(self, episode_dir: Path, meta: dict, capture=None):
        self.dir = Path(episode_dir)
        (self.dir / "imgs").mkdir(parents=True, exist_ok=True)
        self.capture = capture          # () -> {cam: rgb}, 无则不存图
        self.meta = dict(meta)
        self.meta["t_wall_start"] = time.time()
        (self.dir / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        with ExitStack() as files:
            self._f_dec = files.enter_context(open(self.dir / "decisions.jsonl", "a", encoding="utf-8"))
            self._f_traj = files.enter_context(open(self.dir / "trajectory.jsonl", "a", encoding="utf-8"))
            self._files = files.pop_all()
        self._frame = 0
        self._decision_i = 0

    # ---------- 控制级: 由 robot.step_ctrl 的 on_tick 每 0.1s 触发 ----------
    def on_tick(self, model, data, robot) -> None:
        imgs = {}
        if self.capture is not None:
            for cam, rgb in self.capture(data).items():
                p = self.dir / "imgs" / f"{self._frame:06d}_{cam}.jpg"
                p.write_bytes(_jpg_bytes(rgb))
                imgs[cam] = p.relative_to(self.dir).as_posix()
        rec = {
            "t": round(float(data.time), 4),
            "frame": self._frame,
            "imgs": imgs,
            "qpos": [round(float(v), 5) for v in data.qpos],
            "base_pose": [round(float(v), 5) for v in robot.base_pose],
            "arm_qpos": [round(float(v), 5) for v in robot.arm_q],
            "finger": round(float(robot.finger_opening), 4),
            "ctrl": [round(float(v), 4) for v in data.ctrl],
        }
        self._f_traj.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f_traj.flush()
        self._frame += 1

    # ---------- 决策级: 每次工具调用前后 ----------
    def log_decision(self, *, t: float, thought: str, tool: str, args: dict,
                     ok: bool, result: str, img_before: dict | None = None,
                     img_after: dict | None = None, extra: dict | None = None) -> None:
        rec = {
            "i": self._decision_i,
            "t": round(float(t), 4),
            "thought": thought,
            "tool": tool,
            "args": args,
            "ok": bool(ok),
            "result": result,
            "img_before": img_before,
            "img_after": img_after,
        }
        if extra:
            rec.update(extra)
        self._f_dec.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f_dec.flush()
        self._decision_i += 1

    def finish(self, success: bool, info: dict | None = None) -> None:
        self.meta["success"] = bool(success)
        self.meta["frames"] = self._frame
        self.meta["decisions"] = self._decision_i
        self.meta["t_sim_end"] = None
        self.meta["t_wall_end"] = time.time()
        if info:
            self.meta.update(info)
        try:
            (self.dir / "meta.json").write_text(
                json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            self.close()

    def close(self) -> None:
        self._files.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def snapshot_paths(self, data, frame_tag: str) -> dict:
        """决策时刻的双相机快照(供 decisions.jsonl 引用)."""
        if self.capture is None:
            return {}
        out = {}
        for cam, rgb in self.capture(data).items():
            p = self.dir / "imgs" / f"d{self._decision_i:03d}_{frame_tag}_{cam}.jpg"
            p.write_bytes(_jpg_bytes(rgb))
            out[cam] = p.relative_to(self.dir).as_posix()
        return out


def next_episode_dir(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name.split("_")[1]) for p in root.glob("ep_*") if p.is_dir()]
    idx = (max(existing) + 1) if existing else 0
    return root / f"ep_{idx:04d}"
