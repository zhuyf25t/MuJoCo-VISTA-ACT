"""One training objective: arm L1 + gripper CE + balanced base CE + turn L1."""
import h5py
import numpy as np
import torch
from torch.nn import functional as F

from .action_codec import BASE_MODES, action_labels


def training_class_balance(dataset):
    """Count every valid chunk position, respecting episode boundaries and K.

    With one uniformly sampled anchor per frame, target frame t occurs in
    min(K,t+1) chunks. Validation files and padding never contribute.
    """
    counts = np.zeros(len(BASE_MODES), dtype=np.int64)
    for path in dataset.paths:
        with h5py.File(path, "r") as f:
            base, _, _ = action_labels(torch.from_numpy(f["action"][:]))
        multiplicity = np.minimum(np.arange(len(base)) + 1, dataset.cfg.chunk_size)
        np.add.at(counts, base.numpy(), multiplicity)
    present = counts > 0
    if not present.any():
        raise ValueError("Empty training actions")
    weights = np.zeros(len(counts), dtype=np.float64)
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return {"base_modes": list(BASE_MODES), "counts": counts.tolist(), "weights": weights.tolist(),
            "chunk_size": dataset.cfg.chunk_size, "valid_chunk_positions": int(counts.sum()),
            "sampling": "uniform_anchors", "missing_classes": [BASE_MODES[i] for i in np.flatnonzero(~present)]}


def hybrid_loss(heads, normalized_targets, is_pad, stats, base_weights):
    """No argmax in the gradient path. Mask amplitude by the TRUE turn class.

    Weighted CE uses a fixed sample denominator, not a per-batch weight sum:
    otherwise a batch containing only a rare class would cancel its weighting.
    """
    valid = ~is_pad
    if not bool(valid.any()):
        raise ValueError("A batch must contain valid actions")
    target = normalized_targets[valid].float()
    mean = torch.as_tensor(stats["action_mean"], device=target.device, dtype=target.dtype)
    std = torch.as_tensor(stats["action_std"], device=target.device, dtype=target.dtype)
    base, gripper, magnitude = action_labels(target * std + mean, validate=False)
    weights = torch.as_tensor(base_weights, device=target.device, dtype=target.dtype)
    arm = F.l1_loss(heads["arm"][valid].float(), target[:, :2])
    grip = F.cross_entropy(heads["gripper_logits"][valid].float(), gripper)
    base_loss = (F.cross_entropy(heads["base_logits"][valid].float(), base, reduction="none") * weights[base]).mean()
    turns = base >= 3
    predicted_magnitude = heads["turn"][valid].float().squeeze(-1)
    # Index before arithmetic: even NaN padding cannot leak into values/gradients.
    turn = (F.l1_loss(predicted_magnitude[turns], magnitude[turns]) if bool(turns.any())
            else predicted_magnitude[:0].sum())
    parts = {"arm_l1": arm, "gripper_ce": grip, "base_ce": base_loss, "turn_l1": turn}
    return sum(parts.values()), parts, {"valid": int(valid.sum()), "turn": int(turns.sum())}


class LossAverages:
    """Aggregate validation independently of batch boundaries, including no-turn batches."""
    def __init__(self):
        self.sums = {name: 0. for name in ("arm_l1", "gripper_ce", "base_ce", "turn_l1")}
        self.valid = self.turns = 0

    def update(self, parts, counts):
        self.valid += counts["valid"]
        self.turns += counts["turn"]
        for key, value in parts.items():
            self.sums[key] += float(value.detach()) * counts["turn" if key == "turn_l1" else "valid"]

    def result(self, prefix):
        values = {key: value / max(1, self.turns if key == "turn_l1" else self.valid)
                  for key, value in self.sums.items()}
        return {f"{prefix}_loss": sum(values.values()), **{f"{prefix}_{k}": v for k, v in values.items()}}
