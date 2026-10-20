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


def soften_group_weights(weights, counts, power):
    """Temper rare-group influence while preserving mean training weight one."""
    if not np.isfinite(power) or not 0 <= power <= 1:
        raise ValueError("balance power must be in [0, 1]")
    weights, counts = np.asarray(weights, dtype=np.float64), np.asarray(counts)
    if power == 1:
        return weights  # Preserve exact metadata for existing experiments/resume.
    tempered = np.where(weights > 0, weights ** power, 0.)
    total = float((tempered * counts).sum())
    if total <= 0:
        raise ValueError("Need at least one positively weighted training group")
    return tempered * (counts.sum() / total)


def training_class_balance(dataset, power=1.0, first_query_weight=0.0):
    """Count every valid chunk position, respecting episode boundaries and K.

    With one uniformly sampled anchor per frame, target frame t occurs in
    min(K,t+1) chunks. With alpha>0 its objective multiplicity is
    (1-alpha)*min(K,t+1) + alpha*K: every frame is query zero once.
    Validation files and padding never contribute. Original integer occurrence
    counts remain available; group weights use objective counts when enabled.
    """
    if not np.isfinite(first_query_weight) or not 0 <= first_query_weight <= 1:
        raise ValueError("first_query_weight must be in [0, 1]")
    counts = np.zeros(len(BASE_MODES), dtype=np.int64)
    base_groups = np.zeros((len(BASE_MODES), 2), dtype=np.int64)
    grip_groups = np.zeros((2, 2), dtype=np.int64)
    objective_counts = np.zeros_like(counts, dtype=np.float64)
    objective_base = np.zeros_like(base_groups, dtype=np.float64)
    objective_grip = np.zeros_like(grip_groups, dtype=np.float64)
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
        if first_query_weight:
            mass = (1 - first_query_weight) * multiplicity + first_query_weight * dataset.cfg.chunk_size
            np.add.at(objective_counts, base.numpy(), mass)
            np.add.at(objective_base, (base.numpy(), base_switch.long().numpy()), mass)
            np.add.at(objective_grip, (grip.numpy(), grip_switch.long().numpy()), mass)
    balance_counts = objective_counts if first_query_weight else counts
    balance_base = objective_base if first_query_weight else base_groups
    balance_grip = objective_grip if first_query_weight else grip_groups
    present = balance_counts > 0
    if not present.any():
        raise ValueError("Empty training actions")
    weights = np.zeros(len(counts), dtype=np.float64)
    weights[present] = balance_counts.sum() / (present.sum() * balance_counts[present])
    result = {"base_modes": list(BASE_MODES), "counts": counts.tolist(), "weights": weights.tolist(),
            "group_order": ["ordinary", "switch"],
            "base_group_counts": base_groups.tolist(),
            "base_group_weights": soften_group_weights(_group_weights(balance_base, weights), balance_base, power).tolist(),
            "gripper_group_counts": grip_groups.tolist(),
            "gripper_group_weights": soften_group_weights(_group_weights(balance_grip, np.ones(2)), balance_grip, power).tolist(),
            "chunk_size": dataset.cfg.chunk_size, "valid_chunk_positions": int(counts.sum()),
            "sampling": "uniform_anchors", "missing_classes": [BASE_MODES[i] for i in np.flatnonzero(~present)]}
    if first_query_weight:
        result.update(first_query_weight=first_query_weight,
                      query_weight_rule="(1-alpha)+alpha*K*I[j=0]",
                      objective_counts=objective_counts.tolist(),
                      objective_base_group_counts=objective_base.tolist(),
                      objective_gripper_group_counts=objective_grip.tolist(),
                      objective_valid_chunk_positions=float(objective_counts.sum()))
    return result


def hybrid_loss(heads, normalized_targets, is_pad, stats, balance, action_states, first_query_weight=0.0):
    """No argmax in the gradient path. Mask amplitude by the TRUE turn class.

    Weighted CE uses a fixed sample denominator, not a per-batch weight sum:
    otherwise a batch containing only a rare class would cancel its weighting.
    Optional alpha gives position j mass (1-alpha)+alpha*K*I[j=0], normalized
    over valid positions (true-turn positions for amplitude). This is a single
    position-weighted objective, not a mixture of separately normalized means.
    Class/group weights never enter the denominator. Alpha zero preserves the
    original arithmetic and counts metadata exactly.
    """
    if not np.isfinite(first_query_weight) or not 0 <= first_query_weight <= 1:
        raise ValueError("first_query_weight must be in [0, 1]")
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
    if first_query_weight:
        position = torch.full(is_pad.shape, 1 - first_query_weight, device=target.device, dtype=target.dtype)
        position[:, 0] += first_query_weight * is_pad.shape[1]
        position = position[valid]
        active = position > 0
        denominator = position.sum()
        if not bool(denominator > 0):
            raise ValueError("A batch must contain positively weighted valid actions")
        # Drop zero-mass positions before arithmetic, including alpha=1 futures.
        mass = position[active]
        arm_errors = F.l1_loss(heads["arm"][valid][active].float(), target[active, :2], reduction="none").mean(-1)
        arm = (arm_errors * mass).sum() / denominator
        grip_errors = F.cross_entropy(heads["gripper_logits"][valid][active].float(), gripper[active], reduction="none")
        grip = (grip_errors * grip_weights[gripper[active], grip_switch[active].long()] * mass).sum() / denominator
        base_errors = F.cross_entropy(heads["base_logits"][valid][active].float(), base[active], reduction="none")
        base_loss = (base_errors * base_weights[base[active], base_switch[active].long()] * mass).sum() / denominator
        turns = base >= 3
        weighted_turns = turns & active
        predicted_magnitude = heads["turn"][valid].float().squeeze(-1)
        turn_denominator = position[weighted_turns].sum()
        turn = ((F.mse_loss(predicted_magnitude[weighted_turns], magnitude[weighted_turns], reduction="none")
                 * position[weighted_turns]).sum() / turn_denominator
                if bool(weighted_turns.any()) else predicted_magnitude[:0].sum())
        parts = {"arm_l1": arm, "gripper_ce": grip, "base_ce": base_loss, "turn_mse": turn}
        return sum(parts.values()), parts, {"valid": int(valid.sum()), "turn": int(turns.sum()),
                                           "loss_valid": float(denominator), "loss_turn": float(turn_denominator)}
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
        self.loss_valid = self.loss_turns = 0.

    def update(self, parts, counts):
        self.valid += counts["valid"]
        self.turns += counts["turn"]
        valid_mass = counts.get("loss_valid", counts["valid"])
        turn_mass = counts.get("loss_turn", counts["turn"])
        self.loss_valid += valid_mass
        self.loss_turns += turn_mass
        for key, value in parts.items():
            self.sums[key] += float(value.detach()) * (turn_mass if key == "turn_mse" else valid_mass)

    def result(self, prefix):
        values = {key: value / ((self.loss_turns if key == "turn_mse" else self.loss_valid) or 1)
                  for key, value in self.sums.items()}
        return {f"{prefix}_loss": sum(values.values()), **{f"{prefix}_{k}": v for k, v in values.items()}}
