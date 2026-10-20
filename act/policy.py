"""Standalone inference from a checkpoint, without the original training files."""
import numpy as np
import torch

from .action_codec import pack_heads, decode_packet
from .checkpoint import load_checkpoint, model_from_checkpoint, normalized_prediction, deployment_metadata
from .dataset import image_tensor


class ACTPolicy:
    def __init__(self, checkpoint_path, device="cpu"):
        checkpoint = load_checkpoint(checkpoint_path)
        self.deployment = deployment_metadata(checkpoint)
        self.execution_horizon = self.deployment["execution_horizon"]
        self.device = torch.device(device)
        self.model = model_from_checkpoint(checkpoint, self.device)
        self.schema = checkpoint["schema"]
        self.stats = {k: np.asarray(v, np.float32) for k, v in checkpoint["stats"].items()}
        self.reset()

    def reset(self):
        self.step = 0
        self._cached_chunk = None
        self._cache_index = 0

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
        if self._cached_chunk is None or self._cache_index >= self.execution_horizon:
            output = self._predict(images_rgb, measured_state)
            packet = pack_heads(output)[0].cpu()
            if not torch.isfinite(packet).all():
                raise FloatingPointError("Nonfinite policy output")
            self._cached_chunk = decode_packet(packet, self.stats).numpy()
            if not np.isfinite(self._cached_chunk).all():
                raise FloatingPointError("Nonfinite decoded policy output")
            self._cache_index = 0
        selected = self._cached_chunk[self._cache_index].copy()
        self._cache_index += 1
        self.step += 1
        return selected
