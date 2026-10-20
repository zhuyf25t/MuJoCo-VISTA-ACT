"""Save and load checkpoints for the hybrid ACT model."""
import os
from pathlib import Path

import torch

from .action_codec import CONTRACT, decode_heads
from .config import config_from_dict
from .common import validate_schema, validate_stats


# Frozen deployment used for training-time task selection and published best.pt.
# The null ensemble field is descriptive metadata, not a runtime option.
DEFAULT_DEPLOYMENT = {"version": 1, "mode": "chunk_prefix", "execution_horizon": 8,
                      "ensemble_coefficient": None}


def deployment_metadata(checkpoint):
    """Require the current K32/H8 deployment; never infer a missing protocol."""
    saved = checkpoint.get("deployment")
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint requires saved K32/H8 deployment")
    chunk_size = checkpoint.get("config", {}).get("model", {}).get("chunk_size")
    if (type(chunk_size) is not int or chunk_size != 32
            or type(saved.get("version")) is not int
            or type(saved.get("execution_horizon")) is not int
            or any(key not in saved or saved[key] != value for key, value in DEFAULT_DEPLOYMENT.items())):
        raise ValueError("Checkpoint requires saved K32/H8 deployment with ensemble off")
    return dict(DEFAULT_DEPLOYMENT)


def save_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Unsupported checkpoint format; this branch requires a hybrid ACT checkpoint")
    if checkpoint.get("action_codec") != CONTRACT:
        raise ValueError("Unsupported hybrid action contract")
    deployment_metadata(checkpoint)
    return checkpoint


def model_from_checkpoint(checkpoint, device="cpu"):
    from .model import ACT
    model = ACT(config_from_dict(checkpoint["config"]).model, initialize_backbone=False)
    validate_schema(checkpoint["schema"], model.cfg)
    validate_stats(checkpoint["stats"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def normalized_prediction(output, stats):
    physical = decode_heads(output, stats)
    mean = torch.as_tensor(stats["action_mean"], device=physical.device, dtype=physical.dtype)
    std = torch.as_tensor(stats["action_std"], device=physical.device, dtype=physical.dtype)
    return (physical - mean) / std
