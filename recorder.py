"""Durable per-tick recording and whole-episode ACT export.

Raw PNG/JSONL logs survive interruption. Only complete, successful episodes with
valid paired data are published as HDF5; manifest updates are atomic and locked.
The task evaluator lives in the environment, not in this module.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import tempfile
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image

import config


def dataset_schema() -> dict:
    return {
        "version": "mobile_act_v1", "fps": config.CTRL_HZ,
        "observation_timing": "before_action",
        "state_names": ["shoulder_actual", "elbow_actual", "gripper_actual",
                        "base_v_measured", "base_w_measured"],
        "action_names": ["shoulder_target", "elbow_target", "gripper_target",
                         "base_v_cmd", "base_w_cmd"],
        "state_units": ["rad", "rad", "ratio", "m/s", "rad/s"],
        "action_units": ["rad", "rad", "ratio", "m/s", "rad/s"],
        "gripper": {"closed": 0, "open": 1}, "cameras": ["front"],
        "image_shape": [240, 320, 3], "image_color": "RGB",
        "action_semantics": "applied_target_held_for_one_control_interval",
    }


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sync(handle):
    handle.flush()
    os.fsync(handle.fileno())


def _atomic_json(path: Path, value) -> None:
    """A process crash leaves either the old JSON or the complete new JSON."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=".record-", delete=False) as f:
            temporary = Path(f.name)
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            _sync(f)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class FileLock:
    """Local process lock released by the OS even after a crash.

    Keep the file in place: unlinking it can allow different locks for the same
    pathname. Intended for local Windows/Linux filesystems, not NFS.
    """
    def __init__(self, path: Path, timeout: float = 10.0):
        self.path, self.timeout = path, timeout

    def __enter__(self):
        self.file = self.path.open("a+b")
        if self.file.seek(0, os.SEEK_END) == 0:
            self.file.write(b"\0")
            self.file.flush()
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError as error:
                if time.monotonic() >= deadline:
                    self.file.close()
                    raise BlockingIOError(f"记录目录正在使用: {self.path}") from error
                time.sleep(0.05)

    def __exit__(self, *exc):
        self.file.close()


class DatasetStore:
    """One appendable manifest per dataset, one HDF5 per accepted episode."""
    def __init__(self, root: Path):
        import h5py  # Fail before collecting if the export dependency is missing.
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "episodes").mkdir(exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        with self._lock():
            if not self.manifest_path.exists():
                _atomic_json(self.manifest_path, {"schema": dataset_schema(), "episodes": []})
            self._read()

    def _lock(self):
        return FileLock(self.root / ".manifest.lock")

    def _read(self):
        manifest = _read_json(self.manifest_path)
        if manifest.get("schema") != dataset_schema() or not isinstance(manifest.get("episodes"), list):
            raise ValueError("manifest schema 不匹配；请使用新的 --dataset-out 目录")
        return manifest

    def register(self, source: Path, meta: dict) -> str:
        with self._lock():
            manifest = self._read()
            entries = manifest["episodes"]
            number = max((int(e["episode_id"].split("-")[1]) for e in entries), default=-1) + 1
            episode_id = f"episode-{number:04d}"
            entries.append({
                "episode_id": episode_id, "source_dir": str(source.resolve()),
                "source_uid": meta["source_uid"], "task": meta.get("task"), "seed": meta.get("seed"),
                "status": "recording", "success": None, "num_steps": None,
                "h5": None, "selected_for_dataset": False, "reason": "not_finalized",
            })
            _atomic_json(self.manifest_path, manifest)
        return episode_id

    def _update(self, episode_id, **values):
        with self._lock():
            manifest = self._read()
            entry = next(e for e in manifest["episodes"] if e["episode_id"] == episode_id)
            entry.update(values)
            _atomic_json(self.manifest_path, manifest)

    def publish(self, episode_id: str) -> dict:
        """Caller holds the source episode lock. Repeating publication is safe."""
        with self._lock():
            entry = next(e for e in self._read()["episodes"] if e["episode_id"] == episode_id)
        source = Path(entry["source_dir"])
        meta = _read_json(source / "meta.json")
        if meta.get("source_uid") != entry["source_uid"]:
            raise ValueError("原始 episode 身份不匹配")
        if not meta.get("finalized"):
            values = dict(status="interrupted", success=False, num_steps=None,
                          h5=None, reason="process_interrupted")
        elif not meta.get("complete") or not meta.get("success"):
            values = dict(status="rejected" if meta.get("complete") else "interrupted",
                          success=bool(meta.get("success")), num_steps=None,
                          h5=None,
                          reason="task_failed" if meta.get("complete") else "incomplete_episode")
        elif not meta.get("recording_images", True):
            values = dict(status="rejected", success=True, num_steps=None,
                          h5=None, reason="images_disabled")
        else:
            try:
                count = self._export(source, entry, meta)
            except ValueError as error:
                values = dict(status="invalid", success=True, num_steps=None,
                              h5=None, reason=str(error))
            except Exception as error:
                self._update(episode_id, status="export_error", reason=str(error))
                raise
            else:
                values = dict(status="ready", success=True, num_steps=count,
                              h5=f"episodes/{episode_id}.h5", reason=None)
        values["selected_for_dataset"] = values["status"] == "ready"
        self._update(episode_id, **values)
        return {"episode_id": episode_id, **values}

    def _export(self, source, entry, meta):
        import h5py
        count = meta["frames"]
        if not isinstance(count, int) or count < 1:
            raise ValueError("empty_episode")
        if meta.get("schema") != dataset_schema():
            raise ValueError("raw_schema_mismatch")
        destination = self.root / "episodes" / f"{entry['episode_id']}.h5"
        if destination.exists():
            # Recovery after the HDF5 rename but before the manifest replacement.
            with h5py.File(destination, "r") as f:
                if (f.attrs.get("source_uid") != entry["source_uid"]
                        or f.attrs.get("num_steps") != count
                        or f["action"].shape != (count, 5)
                        or f["observations/state"].shape != (count, 5)
                        or f["observations/images/front"].shape != (count, 240, 320, 3)):
                    raise ValueError("existing_hdf5_mismatch")
            return count
        temporary = destination.with_suffix(".tmp.h5")
        try:
            with h5py.File(temporary, "w") as f:
                f.attrs.update(episode_id=entry["episode_id"], source_uid=entry["source_uid"],
                               num_steps=count, fps=config.CTRL_HZ, schema_version="mobile_act_v1")
                f.attrs["schema_json"] = json.dumps(dataset_schema())
                images = f.create_dataset("observations/images/front", (count, 240, 320, 3),
                                          dtype="uint8", chunks=(1, 240, 320, 3),
                                          compression="gzip", compression_opts=1)
                states = f.create_dataset("observations/state", (count, 5), dtype="float32")
                actions = f.create_dataset("action", (count, 5), dtype="float32")
                n, last_end = 0, None
                with (source / "trajectory.jsonl").open(encoding="utf-8") as stream:
                    for line in stream:
                        row = json.loads(line)
                        if n >= count or row.get("frame") != n:
                            raise ValueError("frame_count_mismatch")
                        start, end = float(row["t"]), float(row["t_end"])
                        if (not np.isfinite([start, end]).all()
                                or abs(end - start - config.CTRL_DT) > 1e-6
                                or (last_end is not None and abs(start - last_end) > 1e-6)):
                            raise ValueError("non_contiguous_control_intervals")
                        state, action = np.asarray(row["state"], np.float32), np.asarray(row["action"], np.float32)
                        if (state.shape != (5,) or action.shape != (5,)
                                or not np.isfinite(state).all() or not np.isfinite(action).all()
                                or not 0 <= state[2] <= 1
                                or not np.isclose(action[2], [0, 1], atol=1e-6).any()):
                            raise ValueError("invalid_state_or_action")
                        path = (source / row["imgs"][config.CAM_FRONT]).resolve()
                        if not path.is_relative_to(source.resolve()):
                            raise ValueError("image_path_outside_episode")
                        with Image.open(path) as img:
                            if img.mode != "RGB" or img.size != (320, 240):
                                raise ValueError("expected_320x240_RGB")
                            images[n] = np.asarray(img)
                        states[n], actions[n] = state, action
                        n, last_end = n + 1, end
                if n != count:
                    raise ValueError("frame_count_mismatch")
                terminal = _read_json(source / "terminal.json")
                if abs(float(terminal["t"]) - last_end) > 1e-6:
                    raise ValueError("terminal_time_mismatch")
            with temporary.open("r+b") as handle:
                _sync(handle)
            os.replace(temporary, destination)
        except (KeyError, TypeError) as error:
            raise ValueError("missing_or_invalid_raw_fields") from error
        finally:
            temporary.unlink(missing_ok=True)
        return count

    def recover(self) -> list[dict]:
        """Finish publication or exclude abandoned partial episodes; skip live ones."""
        with self._lock():
            pending = [e for e in self._read()["episodes"]
                       if e["status"] in ("recording", "export_error")]
        results = []
        for entry in pending:
            try:
                with FileLock(Path(entry["source_dir"]) / ".recording.lock", timeout=0):
                    with self._lock():
                        current = next(e for e in self._read()["episodes"]
                                       if e["episode_id"] == entry["episode_id"])
                    if current["status"] in ("recording", "export_error"):
                        results.append(self.publish(entry["episode_id"]))
            except BlockingIOError:
                continue
        return results


class EpisodeRecorder:
    """One raw episode, with pre-action observations paired to completed actions."""
    def __init__(self, episode_dir: Path, meta: dict, capture=None,
                 dataset: DatasetStore | None = None):
        self.dir = Path(episode_dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.capture, self.dataset = capture, dataset
        self._frame = self._decision_i = 0
        self._pending = None
        self._finalized = False
        with ExitStack() as resources:
            resources.enter_context(FileLock(self.dir / ".recording.lock", timeout=0))
            if (self.dir / "meta.json").exists():
                raise FileExistsError(f"不覆盖已有 episode: {self.dir}")
            (self.dir / "imgs").mkdir(exist_ok=True)
            self.meta = {**meta, "source_uid": uuid.uuid4().hex, "schema": dataset_schema(),
                         "t_wall_start": time.time(), "finalized": False,
                         "complete": False, "success": False}
            _atomic_json(self.dir / "meta.json", self.meta)
            self.episode_id = dataset.register(self.dir, self.meta) if dataset else self.dir.name
            self.meta["episode_id"] = self.episode_id
            _atomic_json(self.dir / "meta.json", self.meta)
            self._f_dec = resources.enter_context((self.dir / "decisions.jsonl").open("x", encoding="utf-8"))
            self._f_traj = resources.enter_context((self.dir / "trajectory.jsonl").open("x", encoding="utf-8"))
            self._files = resources.pop_all()

    def _snapshot(self, data, robot, tag):
        # Refresh MuJoCo derived state before rendering; no physical stepping.
        velocity = np.asarray(robot.base_velocity, dtype=float)
        state = np.array([*robot.arm_q, robot.finger_opening, *velocity], dtype=float)
        if state.shape != (5,) or not np.isfinite(state).all():
            raise ValueError("五维实测状态无效")
        return {"t": float(data.time), "imgs": self.snapshot_paths(data, tag),
                "state": state.tolist(), "qpos": data.qpos.tolist(),
                "qvel": data.qvel.tolist(), "base_pose": robot.base_pose.tolist(),
                "arm_qpos": robot.arm_q.tolist(), "finger": float(robot.finger_opening),
                "ctrl": data.ctrl.tolist()}

    def before_tick(self, model, data, robot) -> None:
        if self._pending is not None:
            raise RuntimeError("上一控制拍尚未完成")
        action = np.asarray(robot.hal.applied_action, dtype=float).copy()
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("五维施加动作无效")
        self._pending = {**self._snapshot(data, robot, f"{self._frame:06d}"),
                         "frame": self._frame, "action": action.tolist()}

    def on_tick(self, model, data, robot) -> None:
        """Commit only after the full physical interval has executed."""
        if self._pending is None:
            raise RuntimeError("缺少执行前观测，拒绝将执行后图像当训练输入")
        row = {**self._pending, "t_end": float(data.time)}
        if abs(row["t_end"] - row["t"] - config.CTRL_DT) > 1e-6:
            raise ValueError("控制拍时长不匹配")
        self._f_traj.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        _sync(self._f_traj)
        self._frame += 1
        self._pending = None

    def record_terminal(self, data, robot) -> None:
        _atomic_json(self.dir / "terminal.json", self._snapshot(data, robot, "terminal"))

    def log_decision(self, *, t: float, thought: str, tool: str, args: dict,
                     ok: bool, result: str, img_before: dict | None = None,
                     img_after: dict | None = None, extra: dict | None = None) -> None:
        rec = {"i": self._decision_i, "t": float(t), "thought": thought, "tool": tool,
               "args": args, "ok": bool(ok), "result": result,
               "img_before": img_before, "img_after": img_after, **(extra or {})}
        self._f_dec.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
        _sync(self._f_dec)
        self._decision_i += 1

    def finish(self, success: bool, info: dict | None = None) -> dict | None:
        if self._finalized:
            return None  # Export failure must not overwrite a saved task result.
        info = dict(info or {})
        complete = (not info.get("error") and info.get("stop_code") not in ("window_closed", "interrupted")
                    and self._pending is None and info.get("settling", {}).get("completed", True))
        self.meta.update(info)
        self.meta.update(success=bool(success), complete=bool(complete), finalized=True,
                         frames=self._frame, decisions=self._decision_i, t_wall_end=time.time())
        _atomic_json(self.dir / "meta.json", self.meta)
        self._finalized = True
        # Keep the source lock until publication is complete or context closes.
        self._f_dec.close()
        self._f_traj.close()
        return self.dataset.publish(self.episode_id) if self.dataset else None

    def close(self) -> None:
        self._files.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def snapshot_paths(self, data, frame_tag: str) -> dict:
        if self.capture is None:
            return {}
        out = {}
        for cam, rgb in self.capture(data).items():
            if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                    or (cam == config.CAM_FRONT and rgb.shape != (240, 320, 3))):
                raise ValueError("需要 RGB uint8 图像；车头图须为 320x240")
            p = self.dir / "imgs" / f"{frame_tag}_{cam}.png"
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="PNG")
            with p.open("xb") as f:
                f.write(buffer.getvalue())
                _sync(f)
            out[cam] = p.relative_to(self.dir).as_posix()
        return out


def next_episode_dir(root: Path) -> Path:
    """Atomically reserve a fresh raw directory across CLI processes."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    numbers = [int(m.group(1)) for p in root.iterdir()
               if (m := re.fullmatch(r"ep_(\d+)", p.name))]
    index = max(numbers, default=-1) + 1
    while True:
        path = root / f"ep_{index:04d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="恢复 HDF5 发布并排除崩溃半集；不运行模型")
    parser.add_argument("--dataset-out", type=Path, default=config.ACT_DATA_DIR)
    args = parser.parse_args()
    for result in DatasetStore(args.dataset_out).recover():
        print(json.dumps(result, ensure_ascii=False))
