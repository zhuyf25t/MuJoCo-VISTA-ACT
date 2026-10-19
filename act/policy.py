"""Standalone inference from a checkpoint, without the original training files."""
import numpy as np
import torch

from .action_codec import pack_heads, decode_packet
from .checkpoint import load_checkpoint, model_from_checkpoint, normalized_prediction
from .dataset import image_tensor


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
        self.device = torch.device(device)
        self.model = model_from_checkpoint(checkpoint, self.device)
        self.schema = checkpoint["schema"]
        self.stats = {k: np.asarray(v, np.float32) for k, v in checkpoint["stats"].items()}
        self.ensemble = TemporalEnsembler(ensemble_coefficient) if ensemble_coefficient is not None else None
        self.reset()

    def reset(self):
        self.step = 0
        if self.ensemble:
            self.ensemble.reset()

    @torch.inference_mode()
    def _predict(self, images_rgb, measured_state):
        images = np.asarray(images_rgb)
        cfg = self.model.cfg
        if images.shape != (len(cfg.cameras), *cfg.image_hw, 3):
            raise ValueError("Image shape differs from checkpoint")
        state = np.asarray(measured_state, np.float32)
        if state.shape != (cfg.state_dim,) or not np.isfinite(state).all():
            raise ValueError("Expected finite measured state")
        state = (state - self.stats["state_mean"]) / self.stats["state_std"]
        return self.model(image_tensor(images).unsqueeze(0).to(self.device),
                          torch.from_numpy(state).unsqueeze(0).to(self.device))

    @torch.inference_mode()
    def predict_chunk(self, images_rgb, measured_state):
        output = self._predict(images_rgb, measured_state)
        chunk = normalized_prediction(output, self.stats)[0].cpu().numpy()
        if not np.isfinite(chunk).all():
            raise FloatingPointError("Nonfinite policy output")
        return chunk

    @torch.inference_mode()
    def act(self, images_rgb, measured_state):
        output = self._predict(images_rgb, measured_state)
        chunk = pack_heads(output)[0].cpu().numpy()
        if not np.isfinite(chunk).all():
            raise FloatingPointError("Nonfinite policy output")
        selected = self.ensemble.select(self.step, chunk) if self.ensemble else chunk[0]
        self.step += 1
        return decode_packet(torch.from_numpy(selected), self.stats).numpy()
