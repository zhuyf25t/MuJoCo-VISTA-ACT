"""The v2 head contract, in physical units; no VISTA tool calls are decoded here."""
import torch

BASE_MODES = ("stop", "forward", "back", "left", "right")
DRIVE_SPEED = 0.3  # m/s; distinct from angular speed.
TURN_SPEED = 1.0   # rad/s; the magnitude head predicts a fraction of this range.
CONTRACT = {"version": 2, "base_modes": list(BASE_MODES),
            "gripper_modes": ["closed", "open"], "drive_speed_m_s": DRIVE_SPEED,
            "turn_max_rad_s": TURN_SPEED, "latent": "deterministic",
            "turn_head": "linear_mse", "supervision": "within_class_switch_balance_v1"}


def action_labels(physical, *, validate=True):
    """Use commanded targets, never measured velocities or finger opening."""
    if physical.shape[-1] != 5:
        raise ValueError("Expected five physical action dimensions")
    g, v, w = physical[..., 2], physical[..., 3], physical[..., 4]
    moving, turning = v.abs() > 1e-5, w.abs() > 1e-5
    if validate:
        valid = (torch.isfinite(physical).all()
                 & ((g.abs() < 1e-5) | ((g - 1).abs() < 1e-5)).all()
                 & ((~moving) | ((v.abs() - DRIVE_SPEED).abs() < 1e-5)).all()
                 & (w.abs() <= TURN_SPEED + 1e-5).all()
                 & (~(moving & turning)).all())
        if not bool(valid):
            raise ValueError("Hybrid labels require binary gripper, v=0/±0.3 m/s, |w|<=1 rad/s, and exclusive drive/turn")
    base = torch.zeros_like(v, dtype=torch.long)
    base = torch.where(moving & (v > 0), 1, base)
    base = torch.where(moving & (v < 0), 2, base)
    base = torch.where(turning & (w > 0), 3, base)
    base = torch.where(turning & (w < 0), 4, base)
    return base, (g >= .5).long(), w.abs() / TURN_SPEED


def transition_masks(physical, measured):
    """Identify clear state-to-command switches, using physical units.

    Gripper openings between .4 and .6 and intermediate motion count as ordinary
    unless a different motion component clearly needs to stop. These are measured
    state conditions, not VISTA tool boundaries. Future states are labels only.
    """
    if physical.shape != measured.shape or physical.shape[-1] != 5:
        raise ValueError("Actions and supervision states must have matching [...,5] shapes")
    base, gripper, _ = action_labels(physical, validate=False)
    v, w = measured[..., 3], measured[..., 4]
    moving, turning = v.abs() > .05, w.abs() > .2
    stop_switch = (base == 0) & (moving | turning)
    drive_switch = ((base == 1) | (base == 2)) & (
        (v * physical[..., 3].sign() < .05) | turning)
    turn_switch = (base >= 3) & ((w * physical[..., 4].sign() < .05) | moving)
    grip_switch = ((gripper == 0) & (measured[..., 2] > .6)) | (
        (gripper == 1) & (measured[..., 2] < .4))
    return stop_switch | drive_switch | turn_switch, grip_switch


def pack_heads(heads):
    """Pack continuous arms, probabilities and class-conditioned turn moments.

    Linear temporal fusion is performed on this packet BEFORE choosing a class.
    Turn moments keep unrelated stop/forward/opposite-turn magnitudes out of the
    selected direction's conditional average. Raw amplitudes are fused before
    decode_packet clamps the selected amplitude to the executable range.
    """
    base = heads["base_logits"].float().softmax(-1)
    return torch.cat((heads["arm"].float(), heads["gripper_logits"].float().softmax(-1),
                      base, base[..., 3:5] * heads["turn"].float()), -1)


def decode_packet(packet, stats):
    """An 11-value packet -> five physical actuator targets per 0.1 s tick."""
    mean = torch.as_tensor(stats["action_mean"], device=packet.device, dtype=packet.dtype)
    std = torch.as_tensor(stats["action_std"], device=packet.device, dtype=packet.dtype)
    arm = packet[..., :2] * std[:2] + mean[:2]
    gripper = packet[..., 2:4].argmax(-1).to(packet.dtype)
    base = packet[..., 4:9].argmax(-1)
    v = ((base == 1).to(packet.dtype) - (base == 2).to(packet.dtype)) * DRIVE_SPEED
    turn_prob = packet[..., 7:9]
    magnitude = (packet[..., 9:11] / turn_prob.clamp_min(torch.finfo(packet.dtype).tiny)).clamp(0, 1)
    w = torch.where(base == 3, magnitude[..., 0] * TURN_SPEED,
                    torch.where(base == 4, -magnitude[..., 1] * TURN_SPEED, 0.))
    return torch.cat((arm, gripper[..., None], v[..., None], w[..., None]), -1)


def decode_heads(heads, stats):
    return decode_packet(pack_heads(heads), stats)
