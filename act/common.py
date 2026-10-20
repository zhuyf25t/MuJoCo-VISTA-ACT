"""Portable dataset paths, schema checks and atomic experiment metadata."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".act-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def h5_path(root, relative):
    # Interpret Windows paths on Linux too; source_dir is never used for loading.
    if not isinstance(relative, str) or not relative:
        raise ValueError("Missing relative h5 path")
    posix, windows = PurePosixPath(relative), PureWindowsPath(relative)
    if posix.is_absolute() or windows.drive or windows.root or "\\" in relative or ".." in posix.parts:
        raise ValueError(f"h5 must be a portable relative path: {relative}")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"HDF5 missing or outside dataset root: {relative}")
    return path


def validate_schema(schema, cfg=None):
    expected = {
        "version": "mobile_act_v1", "fps": 10, "observation_timing": "before_action",
        "state_names": ["shoulder_actual", "elbow_actual", "gripper_actual", "base_v_measured", "base_w_measured"],
        "action_names": ["shoulder_target", "elbow_target", "gripper_target", "base_v_cmd", "base_w_cmd"],
        "state_units": ["rad", "rad", "ratio", "m/s", "rad/s"],
        "action_units": ["rad", "rad", "ratio", "m/s", "rad/s"],
        "gripper": {"closed": 0, "open": 1}, "cameras": ["front"],
        "image_shape": [240, 320, 3], "image_color": "RGB",
        "action_semantics": "applied_target_held_for_one_control_interval",
    }
    if schema != expected:
        raise ValueError("Unsupported dataset schema; expected mobile_act_v1 without semantic changes")
    if cfg and (cfg.state_dim != 5 or cfg.action_dim != 5 or cfg.cameras != ("front",)
                or cfg.image_hw != (240, 320)):
        raise ValueError("Model configuration does not match dataset schema")


def validate_stats(stats, state_dim=5, action_dim=5):
    for kind, dim in (("state", state_dim), ("action", action_dim)):
        for suffix in ("mean", "std"):
            array = np.asarray(stats[f"{kind}_{suffix}"], dtype=np.float32)
            if array.shape != (dim,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid {kind}_{suffix}")
            if suffix == "std" and np.any(array <= 0):
                raise ValueError("Normalization standard deviations must be positive")


def load_prepared(experiment, root, cfg=None, verify_hashes=True):
    experiment = Path(experiment)
    manifest = read_json(experiment / "manifest.json")
    splits = read_json(experiment / "splits.json")
    stats = read_json(experiment / "stats.json")
    metadata = read_json(experiment / "prepared.json")
    if metadata.get("format_version") != 2 or metadata.get("training_mode") != "all_demonstrations":
        raise ValueError("Prepared data requires v2/all_demonstrations")
    for name, value in (("manifest", manifest), ("splits", splits), ("stats", stats)):
        if digest(value) != metadata[f"{name}_sha256"]:
            raise ValueError(f"Prepared {name} has changed; create a new experiment")
    validate_schema(manifest["schema"], cfg)
    validate_stats(stats)
    entries = {e["episode_id"]: e for e in manifest["episodes"]}
    if len(entries) != len(manifest["episodes"]):
        raise ValueError("Duplicate episode ids")
    if (set(splits) != {"train", "val"} or not isinstance(splits["train"], list)
            or not splits["train"] or splits["val"] != []
            or len(set(splits["train"])) != len(splits["train"]) or set(splits["train"]) != set(entries)):
        raise ValueError("Prepared data requires all demonstrations in train and val=[]")
    for entry in entries.values():
        if not (entry["status"] == "ready" and entry["selected_for_dataset"] is True and entry["success"] is True):
            raise ValueError("Unselected episode in frozen manifest")
        path = h5_path(root, entry["h5"])
        if verify_hashes and file_digest(path) != metadata["h5_sha256"][entry["episode_id"]]:
            raise ValueError(f"Dataset file changed: {path}")
    return manifest, splits, stats, metadata
