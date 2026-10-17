"""Standalone inference from a checkpoint, without the original training files."""
import numpy as np
import torch

from .common import validate_schema, validate_stats
from .config import config_from_dict
from .dataset import image_tensor
from .model import ACT
from .train import load_checkpoint


class TemporalEnsembler:
    """Combine predictions for the SAME control tick; positive m favors older chunks."""
    def __init__(self, coefficient=0.01):
        if not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError("coefficient must be finite and nonnegative")
        self.coefficient = coefficient
        self.reset()

    def reset(self):
        self.chunks = []

    def select(self, step, chunk):
        chunk = np.asarray(chunk)
        if chunk.ndim != 2 or len(chunk) == 0 or not np.isfinite(chunk).all():
            raise ValueError("Expected finite [K,action_dim] chunk")
        if self.chunks and step <= self.chunks[-1][0]:
            raise ValueError("Control steps must increase; reset between episodes")
        self.chunks.append((step, chunk.copy()))
        self.chunks = [(start, array) for start, array in self.chunks if step - start < len(array)]
        values = np.stack([array[step - start] for start, array in self.chunks])
        weights = np.exp(-self.coefficient * np.arange(len(values)))
        return (values * (weights / weights.sum())[:, None]).sum(0)


class ACTPolicy:
    def __init__(self, checkpoint_path, device="cpu", ensemble_coefficient=None):
        checkpoint = load_checkpoint(checkpoint_path)
        cfg = config_from_dict(checkpoint["config"])
        validate_schema(checkpoint["schema"], cfg.model)
        validate_stats(checkpoint["stats"])
        self.device = torch.device(device)
        # All CNN weights are inside the checkpoint. Never download at inference.
        self.model = ACT(cfg.model, initialize_backbone=False).to(self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.schema = checkpoint["schema"]
        self.stats = {k: np.asarray(v, np.float32) for k, v in checkpoint["stats"].items()}
        self.ensemble = TemporalEnsembler(ensemble_coefficient) if ensemble_coefficient is not None else None
        self.reset()

    def reset(self):
        self.step = 0
        if self.ensemble:
            self.ensemble.reset()

    @torch.inference_mode()
    def predict_chunk(self, images_rgb, measured_state):
        images = np.asarray(images_rgb)
        cfg = self.model.cfg
        if images.shape != (len(cfg.cameras), *cfg.image_hw, 3):
            raise ValueError("Image shape differs from checkpoint")
        state = np.asarray(measured_state, np.float32)
        if state.shape != (cfg.state_dim,) or not np.isfinite(state).all():
            raise ValueError("Expected finite measured state")
        state = (state - self.stats["state_mean"]) / self.stats["state_std"]
        prediction, _, _ = self.model(image_tensor(images).unsqueeze(0).to(self.device),
                                     torch.from_numpy(state).unsqueeze(0).to(self.device))
        chunk = prediction[0].cpu().numpy()
        if not np.isfinite(chunk).all():
            raise FloatingPointError("Nonfinite policy output")
        return chunk  # Normalized; ensemble before conversion to physical units.

    def act(self, images_rgb, measured_state):
        chunk = self.predict_chunk(images_rgb, measured_state)
        action = self.ensemble.select(self.step, chunk) if self.ensemble else chunk[0]
        self.step += 1
        return action * self.stats["action_std"] + self.stats["action_mean"]
