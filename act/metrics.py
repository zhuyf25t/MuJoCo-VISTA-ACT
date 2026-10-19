"""Motion-specific validation: global MAE can hide an always-straight policy."""
import torch

from .action_codec import BASE_MODES, action_labels


class MotionMetrics:
    """Accumulate physical-unit metrics; padding never contributes.

    Turn recall requires the correct direction AND |w| > 0.2 rad/s.
    A start opportunity has a turning target while measured |w| < 0.05.
    These are observation-based opportunities, not high-level tool boundaries.
    """
    def __init__(self, device):
        self.sums = torch.zeros(23, dtype=torch.float64, device=device)
        self.base_confusion = torch.zeros(5, 5, dtype=torch.int64, device=device)
        self.first_base_confusion = torch.zeros_like(self.base_confusion)
        self.gripper_confusion = torch.zeros(2, 2, dtype=torch.int64, device=device)
        self.opening = torch.zeros(2, dtype=torch.int64, device=device)
        self.drive_starts = torch.zeros(2, 2, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, prediction, target, is_pad, state):
        valid = ~is_pad
        w, wanted = prediction[..., 4], target[..., 4]
        active = valid & (wanted.abs() > .2)
        still = valid & (wanted.abs() < .05)
        predicted_active = w.abs() > .2
        correct = predicted_active & (w.sign() == wanted.sign())
        first = active[:, 0]
        starts = first & (state[:, 4].abs() < .05)
        left_starts = starts & (wanted[:, 0] > .2)
        right_starts = starts & (wanted[:, 0] < -.2)
        stops = still[:, 0] & (state[:, 4].abs() > .2)
        closing = valid[:, 0] & (target[:, 0, 2] < .5) & (state[:, 2] > .6)
        values = [active.sum(), (correct & active).sum(),
                  torch.where(active, (w - wanted).abs(), 0).sum(),
                  still.sum(), (predicted_active & still).sum(),
                  torch.where(valid, wanted.abs(), 0).sum(), valid.sum(),
                  first.sum(), (correct[:, 0] & first).sum(),
                  starts.sum(), (correct[:, 0] & starts).sum(),
                  closing.sum(), ((prediction[:, 0, 2] <= .4) & closing).sum(),
                  torch.where(valid, (w - wanted).abs(), 0).sum(),
                  still[:, 0].sum(), (predicted_active[:, 0] & still[:, 0]).sum(),
                  stops.sum(), ((w[:, 0].abs() < .05) & stops).sum(),
                  torch.where(stops, w[:, 0].abs(), 0).sum(),
                  left_starts.sum(), (correct[:, 0] & left_starts).sum(),
                  right_starts.sum(), (correct[:, 0] & right_starts).sum()]
        self.sums += torch.stack(values).double()
        actual_base, actual_grip, _ = action_labels(target[valid], validate=False)
        predicted_base, predicted_grip, _ = action_labels(prediction[valid], validate=False)
        self.base_confusion += torch.bincount(actual_base * 5 + predicted_base, minlength=25).reshape(5, 5)
        self.gripper_confusion += torch.bincount(actual_grip * 2 + predicted_grip, minlength=4).reshape(2, 2)
        first_valid = valid[:, 0]
        wanted_base, _, _ = action_labels(target[:, 0], validate=False)
        got_base, _, _ = action_labels(prediction[:, 0], validate=False)
        self.first_base_confusion += torch.bincount(
            (wanted_base * 5 + got_base)[first_valid], minlength=25).reshape(5, 5)
        opening = first_valid & (target[:, 0, 2] > .5) & (state[:, 2] < .4)
        self.opening += torch.stack((opening.sum(), (opening & (prediction[:, 0, 2] >= .6)).sum()))
        stopped = (state[:, 3].abs() < .05) & (state[:, 4].abs() < .05)
        for index, mode in enumerate((1, 2)):
            starts = first_valid & stopped & (wanted_base == mode)
            self.drive_starts[index] += torch.stack((starts.sum(), (starts & (got_base == mode)).sum()))

    def result(self):
        s = self.sums.cpu().tolist()
        def ratio(n, d):
            return s[n] / s[d] if s[d] else None
        result = {
            "val_turn_active_count": int(s[0]),
            "val_turn_recall": ratio(1, 0),
            "val_turn_mae_active": ratio(2, 0),
            "val_turn_false_positive_rate": ratio(4, 3),
            "val_zero_turn_mae": ratio(5, 6),
            "val_first_turn_count": int(s[7]),
            "val_first_turn_recall": ratio(8, 7),
            "val_first_turn_start_count": int(s[9]),
            "val_first_turn_start_recall": ratio(10, 9),
            "val_first_close_from_open_count": int(s[11]),
            "val_first_close_from_open_recall": ratio(12, 11),
            "val_turn_mae_vs_zero": s[13] / s[5] if s[5] else None,
            "val_first_turn_false_positive_rate": ratio(15, 14),
            "val_first_turn_stop_count": int(s[16]),
            "val_first_turn_stop_recall": ratio(17, 16),
            "val_first_turn_stop_mae": ratio(18, 16),
            "val_first_turn_left_start_count": int(s[19]),
            "val_first_turn_left_start_recall": ratio(20, 19),
            "val_first_turn_right_start_count": int(s[21]),
            "val_first_turn_right_start_recall": ratio(22, 21),
        }
        for prefix, matrix, names in (("base", self.base_confusion, BASE_MODES),
                                      ("first_base", self.first_base_confusion, BASE_MODES),
                                      ("gripper", self.gripper_confusion, ("closed", "open"))):
            matrix = matrix.cpu().tolist()
            recalls = {name: matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else None
                       for i, name in enumerate(names)}
            present = [v for v in recalls.values() if v is not None]
            result.update({f"val_{prefix}_confusion": matrix, f"val_{prefix}_recall": recalls,
                           f"val_{prefix}_macro_recall": sum(present) / len(present) if present else None})
        opening_count, opening_correct = self.opening.cpu().tolist()
        result["val_first_open_from_closed_count"] = opening_count
        result["val_first_open_from_closed_recall"] = opening_correct / opening_count if opening_count else None
        for name, (count, correct) in zip(("forward", "back"), self.drive_starts.cpu().tolist()):
            result[f"val_first_{name}_start_count"] = count
            result[f"val_first_{name}_start_recall"] = correct / count if count else None
        return result
