"""Save and load checkpoints for the hybrid ACT model."""
import os
from pathlib import Path

import torch

from .action_codec import CONTRACT, decode_heads
from .config import config_from_dict
from .common import validate_schema, validate_stats


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
