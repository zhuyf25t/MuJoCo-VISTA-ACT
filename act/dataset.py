"""One sample per control tick, with overlapping future action chunks."""
from bisect import bisect_right

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import h5_path, validate_stats


def image_tensor(images):
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Expected uint8 RGB [camera,height,width,3]")
    return torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).float() / 255.0


class EpisodeDataset(Dataset):
    def __init__(self, root, entries, stats, cfg):
        validate_stats(stats, cfg.state_dim, cfg.action_dim)
        self.cfg = cfg
        self.stats = {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}
        self.paths, lengths = [], []
        for entry in entries:
            path = h5_path(root, entry["h5"])
            with h5py.File(path, "r") as f:
                n = entry["num_steps"]
                if n < 1 or f["action"].shape != (n, cfg.action_dim) or f["observations/state"].shape != (n, cfg.state_dim):
                    raise ValueError(f"Invalid episode shape: {path}")
                for camera in cfg.cameras:
                    if f[f"observations/images/{camera}"].shape != (n, *cfg.image_hw, 3):
                        raise ValueError(f"Image shape mismatch: {path}")
            self.paths.append(path)
            lengths.append(n)
        self.ends = np.cumsum(lengths).tolist()

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = bisect_right(self.ends, index)
        t = index - (self.ends[ep - 1] if ep else 0)
        # Each worker opens its own file; no h5py handles cross process boundaries.
        with h5py.File(self.paths[ep], "r") as f:
            images = np.stack([f[f"observations/images/{camera}"][t] for camera in self.cfg.cameras])
            state = f["observations/state"][t]
            actions = f["action"][t:t + self.cfg.chunk_size]
            # Supervision only: o_{t+j} classifies a_{t+j} as a switch or continuation.
            # The policy still receives only images[t] and state[t].
            action_states = f["observations/state"][t:t + self.cfg.chunk_size]
        state = (state - self.stats["state_mean"]) / self.stats["state_std"]
        actions = (actions - self.stats["action_mean"]) / self.stats["action_std"]
        padded = np.zeros((self.cfg.chunk_size, self.cfg.action_dim), np.float32)
        padded[:len(actions)] = actions
        padded_states = np.zeros((self.cfg.chunk_size, self.cfg.state_dim), np.float32)
        padded_states[:len(action_states)] = action_states
        return {"images": image_tensor(images), "state": torch.from_numpy(state),
                "actions": torch.from_numpy(padded),
                "action_states": torch.from_numpy(padded_states),  # Physical units, never model input.
                "is_pad": torch.arange(self.cfg.chunk_size) >= len(actions)}
