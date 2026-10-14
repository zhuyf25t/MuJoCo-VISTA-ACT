"""Immutable image/state snapshots, fixed notes and append-only JSONL logs."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
from copy import deepcopy
from numbers import Real
from pathlib import Path

import numpy as np
from langchain_core.messages import BaseMessage, message_to_dict
from PIL import Image

from .settings import VistaSettings


def _json_safe(value):
    # A malformed tool argument may contain NaN in an injected/test model reply.
    # Preserve that fact in diagnostics without writing invalid JSONL.
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_number": str(value)}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_text(path: Path, text: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=".vista-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class VistaStorage:
    def __init__(self, settings: VistaSettings):
        self.settings = settings
        self.root = settings.output_root
        self.episode_dir: Path | None = None
        self.frames: dict[str, dict] = {}
        self._frame_hashes: dict[str, str] = {}
        self.current_frame_id: str | None = None
        self.batch_count = 0

    def begin_episode(self, collection_dir: Path, task_text: str) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            guide = self.root / "guide.md"
            if not guide.exists():
                with guide.open("x", encoding="utf-8") as handle:
                    handle.write("暂无经过验证的经验。\n")
            numbers = [int(match.group(1)) for path in self.root.iterdir()
                       if path.is_dir() and (match := re.fullmatch(r"ep_(\d+)", path.name))]
            episode = self.root / f"ep_{max(numbers, default=-1) + 1:04d}"
            episode.mkdir(exist_ok=False)
            self.episode_dir = episode.resolve()
            (episode / "frames").mkdir()
            (episode / "working.md").write_text(
                "当前目标：根据任务和车头图制定计划。\n当前假设：待观察。\n"
                "下一步：检查初始图。\n预期：待提出动作。\n"
                "未决疑点：暂无。每项记录待验证状态或矛盾、关联帧、阻塞的计划、下一检查和解决标准。\n"
                "已解决疑点与直接证据：暂无。\n相关帧编号：暂无。\n",
                encoding="utf-8")
            for name in ("frame_index", "actions", "calls", "messages"):
                (episode / f"{name}.jsonl").touch(exist_ok=False)
        except OSError:
            raise RuntimeError("MEMORY_IO_ERROR: 无法创建 VISTA 图片、笔记或日志目录") from None
        self.frames = {}
        self._frame_hashes = {}
        self.current_frame_id = None
        self.batch_count = 0
        self.read_memory("guide")
        self.read_memory("working")
        self.event("episode_start", collection_dir=str(Path(collection_dir).resolve()), task=task_text)

    def _episode(self) -> Path:
        if self.episode_dir is None:
            raise RuntimeError("VISTA 尚未初始化 episode")
        return self.episode_dir

    def append(self, name: str, record: dict) -> None:
        if name not in ("frame_index", "actions", "calls", "messages"):
            raise RuntimeError("VISTA 未知日志类型")
        path = self._episode() / f"{name}.jsonl"
        try:
            if not path.is_file() or not path.resolve().is_relative_to(self._episode()):
                raise OSError("log unavailable")
            line = json.dumps(_json_safe(record), ensure_ascii=False, allow_nan=False)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
        except (OSError, TypeError, ValueError):
            raise RuntimeError(f"STORAGE_IO_ERROR: 无法写入 VISTA {name}.jsonl") from None

    def event(self, event: str, **fields) -> None:
        self.append("calls", {"event": event, **fields})

    def diagnostic(self, event: str, original: BaseException, **fields) -> None:
        """Logging a secondary error must never mask the original failure."""
        try:
            self.event(event, error=str(original), **fields)
        except Exception as error:
            original.add_note(f"VISTA 诊断记录也未能保存：{type(error).__name__}")

    def log_message(self, message: BaseMessage) -> None:
        data = message_to_dict(message)
        content = data["data"]["content"]
        references = iter(message.additional_kwargs.get("vista_images", []))
        if isinstance(content, list):
            for index, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "image_url":
                    reference = next(references, None)
                    if reference is None:
                        raise RuntimeError("VISTA 图片消息缺少来源记录")
                    content[index] = {"type": "image_reference", **reference}
        self.append("messages", data)

    @staticmethod
    def front_rgb(obs: dict) -> np.ndarray:
        rgb = obs.get("images", {}).get("front") if isinstance(obs, dict) else None
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
                or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1):
            raise RuntimeError("FRAME_UNAVAILABLE: VISTA 需要 images.front 的非空 uint8 RGB 原图")
        return rgb

    @staticmethod
    def observation_state(obs: dict) -> dict:
        """Copy only public measurements; never forward arbitrary environment fields."""
        def finite(value):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError
            value = float(value)
            if not math.isfinite(value):
                raise ValueError
            return value

        try:
            state = obs["observation"]
            public = {key: finite(state[key]) for key in (
                "shoulder_rad", "elbow_rad", "gripper_opening", "base_v_mps", "base_w_radps")}
            if not 0 <= public["gripper_opening"] <= 1:
                raise ValueError
            return {"timestamp_s": finite(obs["timestamp_s"]), "observation": public}
        except (KeyError, TypeError, ValueError, OverflowError):
            raise RuntimeError("OBSERVATION_UNAVAILABLE: 需要有效采样时间和五维实测状态，夹爪开度须在 [0,1]") from None

    def archive(self, obs: dict, source_batch_id: int | None) -> dict:
        rgb = self.front_rgb(obs)
        snapshot = self.observation_state(obs)
        frame_id = f"f{len(self.frames):06d}"
        path = self._episode() / "frames" / f"{frame_id}.png"
        if not path.parent.resolve().is_relative_to(self._episode()):
            raise RuntimeError("FRAME_UNAVAILABLE: 图片目录已经移出本 episode")
        try:
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="PNG")
            contents = buffer.getvalue()
            # Exclusive creation: a published original is never overwritten.
            with path.open("xb") as handle:
                handle.write(contents)
                handle.flush()
            info = {"episode_id": self._episode().name, "frame_id": frame_id,
                    "camera": "front", "path": f"frames/{frame_id}.png",
                    "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                    "source_batch_id": source_batch_id, **snapshot}
            self.append("frame_index", info)
        except (OSError, ValueError):
            raise RuntimeError("FRAME_UNAVAILABLE: 无法保存 VISTA 原始 PNG，未发布该帧") from None
        self.frames[frame_id] = info
        self._frame_hashes[frame_id] = hashlib.sha256(contents).hexdigest()
        self.current_frame_id = frame_id
        return deepcopy(info)

    def frame_info(self, frame_id: str) -> dict:
        if frame_id not in self.frames:
            raise KeyError(frame_id)
        return deepcopy(self.frames[frame_id])

    def load_image(self, frame_id: str) -> Image.Image:
        info = self.frame_info(frame_id)
        frames_dir = self._episode() / "frames"
        path = self._episode() / info["path"]
        try:
            if (not frames_dir.resolve().is_relative_to(self._episode())
                    or not path.resolve().is_relative_to(frames_dir.resolve())):
                raise OSError("frame path changed")
            contents = path.read_bytes()
            if hashlib.sha256(contents).hexdigest() != self._frame_hashes[frame_id]:
                raise OSError("published frame was modified")
            with Image.open(io.BytesIO(contents)) as image:
                image.load()
                if image.mode != "RGB" or image.size != (info["width"], info["height"]):
                    raise OSError("frame metadata mismatch")
                return image.copy()
        except (OSError, ValueError):
            raise RuntimeError(f"FRAME_UNAVAILABLE: 原图 {frame_id} 丢失、损坏或被修改") from None

    def check_observation(self, obs: dict) -> None:
        rgb = self.front_rgb(obs)
        snapshot = self.observation_state(obs)
        saved = self.frame_info(self.current_frame_id)
        if (any(snapshot[key] != saved[key] for key in snapshot)
                or not np.array_equal(rgb, np.asarray(self.load_image(self.current_frame_id)))):
            raise RuntimeError("VISTA 当前观察与已接收的批次最终图或实测状态不一致，请先接回真实批次结果")

    def _memory_path(self, kind: str) -> Path:
        if kind not in ("guide", "working"):
            raise RuntimeError("VISTA 未知笔记类型")
        parent = self.root if kind == "guide" else self._episode()
        path = parent / f"{kind}.md"
        if not path.is_file() or not path.resolve().is_relative_to(parent):
            raise RuntimeError(f"MEMORY_IO_ERROR: {kind}.md 丢失或路径已改变")
        return path

    def read_memory(self, kind: str) -> dict:
        try:
            content = self._memory_path(kind).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise RuntimeError(f"MEMORY_IO_ERROR: 无法读取 {kind}.md") from None
        maximum = self.settings.max_guide_chars if kind == "guide" else self.settings.max_working_chars
        if not content.strip() or len(content) > maximum:
            raise RuntimeError(f"MEMORY_IO_ERROR: {kind}.md 为空或超出 {maximum} 字符")
        return {"content": content, "char_count": len(content),
                "scope": "environment" if kind == "guide" else "episode"}

    def write_memory(self, kind: str, content: str) -> dict:
        # Tool validation has already checked the full replacement text.
        try:
            _atomic_text(self._memory_path(kind), content)
        except (OSError, UnicodeError):
            raise RuntimeError(f"MEMORY_IO_ERROR: 无法完整写入 {kind}.md") from None
        return {**self.read_memory(kind), "written": True}

    def append_batch(self, record: dict) -> None:
        if record["batch_id"] != self.batch_count + 1:
            raise RuntimeError("VISTA 批次编号不连续或重复结算")
        self.append("actions", record)
        self.batch_count += 1

    def history(self, start_batch=None, end_batch=None, limit=10) -> dict:
        try:
            path = self._episode() / "actions.jsonl"
            if not path.resolve().is_relative_to(self._episode()):
                raise ValueError("history path changed")
            entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            required = {"batch_id", "tool_call_id", "observation", "actions", "expectation", "before_frame_id",
                        "after_frame_id", "execution_status", "action_results"}
            if len(entries) != self.batch_count:
                raise ValueError("history count mismatch")
            for index, entry in enumerate(entries, 1):
                if (not isinstance(entry, dict) or not required <= entry.keys()
                        or type(entry["batch_id"]) is not int or entry["batch_id"] != index
                        or not isinstance(entry["actions"], list)
                        or not isinstance(entry["action_results"], list)
                        or len(entry["actions"]) != len(entry["action_results"])):
                    raise ValueError("invalid batch record")
                if entry["observation"] != self.frames[entry["before_frame_id"]]["observation"]:
                    raise ValueError("batch observation does not match its input frame")
            filtered = [{key: entry[key] for key in required} for entry in entries
                        if (start_batch is None or entry["batch_id"] >= start_batch)
                        and (end_batch is None or entry["batch_id"] <= end_batch)]
        except (OSError, UnicodeError, ValueError, TypeError, KeyError):
            raise RuntimeError("HISTORY_UNAVAILABLE: 批次历史丢失或损坏") from None
        selected = filtered[-limit:]
        return {"batches": selected, "truncated": len(filtered) > limit,
                "returned_range": ([selected[0]["batch_id"], selected[-1]["batch_id"]]
                                   if selected else None)}

    def outcome(self, outcome: dict) -> None:
        try:
            _atomic_text(self._episode() / "outcome.json",
                         json.dumps(outcome, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        except (OSError, ValueError, TypeError):
            raise RuntimeError("STORAGE_IO_ERROR: 无法保存 VISTA episode 结局") from None
