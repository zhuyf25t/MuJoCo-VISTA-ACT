"""Arm L1 + within-class switch-balanced gripper/base CE + raw turn MSE."""
import h5py
import numpy as np
import torch
from torch.nn import functional as F

from .action_codec import BASE_MODES, action_labels, transition_masks


def _group_weights(counts, class_weights):
    """Preserve each class's total weight; split it equally over present groups.

    A subgroup absent in training retains its class weight for validation rather
    than silently contributing zero loss when it appears in held-out scenes.
    """
    weights = np.repeat(np.asarray(class_weights)[:, None], 2, axis=1)
    for c, row in enumerate(counts):
        present = row > 0
        if present.any():
            weights[c, present] *= row.sum() / (present.sum() * row[present])
    return weights


def training_class_balance(dataset):
    """Count every valid chunk position, respecting episode boundaries and K.

    With one uniformly sampled anchor per frame, target frame t occurs in
    min(K,t+1) chunks. Validation files and padding never contribute.
    """
    counts = np.zeros(len(BASE_MODES), dtype=np.int64)
    base_groups = np.zeros((len(BASE_MODES), 2), dtype=np.int64)
    grip_groups = np.zeros((2, 2), dtype=np.int64)
    for path in dataset.paths:
        with h5py.File(path, "r") as f:
            actions = torch.from_numpy(f["action"][:])
            states = torch.from_numpy(f["observations/state"][:])
        if not torch.isfinite(states).all():
            raise ValueError(f"Nonfinite supervision states: {path}")
        base, grip, _ = action_labels(actions)
        base_switch, grip_switch = transition_masks(actions, states)
        multiplicity = np.minimum(np.arange(len(base)) + 1, dataset.cfg.chunk_size)
        np.add.at(counts, base.numpy(), multiplicity)
        np.add.at(base_groups, (base.numpy(), base_switch.long().numpy()), multiplicity)
        np.add.at(grip_groups, (grip.numpy(), grip_switch.long().numpy()), multiplicity)
    present = counts > 0
    if not present.any():
        raise ValueError("Empty training actions")
    weights = np.zeros(len(counts), dtype=np.float64)
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return {"base_modes": list(BASE_MODES), "counts": counts.tolist(), "weights": weights.tolist(),
            "group_order": ["ordinary", "switch"],
            "base_group_counts": base_groups.tolist(),
            "base_group_weights": _group_weights(base_groups, weights).tolist(),
            "gripper_group_counts": grip_groups.tolist(),
            "gripper_group_weights": _group_weights(grip_groups, np.ones(2)).tolist(),
            "chunk_size": dataset.cfg.chunk_size, "valid_chunk_positions": int(counts.sum()),
            "sampling": "uniform_anchors", "missing_classes": [BASE_MODES[i] for i in np.flatnonzero(~present)]}


def hybrid_loss(heads, normalized_targets, is_pad, stats, balance, action_states):
    """No argmax in the gradient path. Mask amplitude by the TRUE turn class.

    Weighted CE uses a fixed sample denominator, not a per-batch weight sum:
    otherwise a batch containing only a rare class would cancel its weighting.
    """
    valid = ~is_pad
    if not bool(valid.any()):
        raise ValueError("A batch must contain valid actions")
    target = normalized_targets[valid].float()
    measured = action_states[valid].float()
    mean = torch.as_tensor(stats["action_mean"], device=target.device, dtype=target.dtype)
    std = torch.as_tensor(stats["action_std"], device=target.device, dtype=target.dtype)
    physical = target * std + mean
    base, gripper, magnitude = action_labels(physical, validate=False)
    base_switch, grip_switch = transition_masks(physical, measured)
    base_weights = torch.as_tensor(balance["base_group_weights"], device=target.device, dtype=target.dtype)
    grip_weights = torch.as_tensor(balance["gripper_group_weights"], device=target.device, dtype=target.dtype)
    arm = F.l1_loss(heads["arm"][valid].float(), target[:, :2])
    grip = (F.cross_entropy(heads["gripper_logits"][valid].float(), gripper, reduction="none")
            * grip_weights[gripper, grip_switch.long()]).mean()
    base_loss = (F.cross_entropy(heads["base_logits"][valid].float(), base, reduction="none")
                 * base_weights[base, base_switch.long()]).mean()
    turns = base >= 3
    predicted_magnitude = heads["turn"][valid].float().squeeze(-1)
    # Index before arithmetic: even NaN padding cannot leak into values/gradients.
    turn = (F.mse_loss(predicted_magnitude[turns], magnitude[turns]) if bool(turns.any())
            else predicted_magnitude[:0].sum())
    parts = {"arm_l1": arm, "gripper_ce": grip, "base_ce": base_loss, "turn_mse": turn}
    return sum(parts.values()), parts, {"valid": int(valid.sum()), "turn": int(turns.sum())}


class LossAverages:
    """Aggregate validation independently of batch boundaries, including no-turn batches."""
    def __init__(self):
        self.sums = {name: 0. for name in ("arm_l1", "gripper_ce", "base_ce", "turn_mse")}
        self.valid = self.turns = 0

    def update(self, parts, counts):
        self.valid += counts["valid"]
        self.turns += counts["turn"]
        for key, value in parts.items():
            self.sums[key] += float(value.detach()) * counts["turn" if key == "turn_mse" else "valid"]

    def result(self, prefix):
        values = {key: value / max(1, self.turns if key == "turn_mse" else self.valid)
                  for key, value in self.sums.items()}
        return {f"{prefix}_loss": sum(values.values()), **{f"{prefix}_{k}": v for k, v in values.items()}}
