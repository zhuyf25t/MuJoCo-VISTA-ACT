"""Head/label units, weighted gradients, masked losses and checkpoint validation."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from act.action_codec import action_labels, decode_heads, decode_packet, pack_heads, transition_masks
from act.losses import hybrid_loss, training_class_balance, LossAverages
from act.policy import TemporalEnsembler

STATS = {"action_mean": [1., 2., .4, .05, .1], "action_std": [.5, .7, .2, .3, .4],
         "state_mean": [0.] * 5, "state_std": [1.] * 5}


def heads(n=5):
    return {"arm": torch.zeros(1, n, 2, requires_grad=True),
            "gripper_logits": torch.zeros(1, n, 2, requires_grad=True),
            "base_logits": torch.zeros(1, n, 5, requires_grad=True),
            "turn": torch.full((1, n, 1), .8, requires_grad=True)}


def physical_targets():
    return torch.tensor([[[1., 2., 1., 0., 0.], [1., 2., 0., .3, 0.],
                          [1., 2., 1., -.3, 0.], [1., 2., 0., 0., .7],
                          [1., 2., 1., 0., -.9]]])


def normalized(target):
    return (target - torch.tensor(STATS['action_mean'])) / torch.tensor(STATS['action_std'])


def balance(weights=None):
    return {'base_group_weights': [[w, w] for w in (weights or [1.] * 5)],
            'gripper_group_weights': [[1., 1.], [1., 1.]]}


def test_codec_distinguishes_units_signs_and_binary_targets():
    target = physical_targets()
    base, grip, magnitude = action_labels(target)
    assert base.tolist() == [[0, 1, 2, 3, 4]]
    assert grip.tolist() == [[1, 0, 1, 0, 1]]
    output = heads()
    output['base_logits'] = torch.eye(5)[None] * 20
    output['gripper_logits'] = torch.nn.functional.one_hot(grip, 2).float() * 20
    output['turn'] = magnitude[..., None]
    torch.testing.assert_close(decode_heads(output, STATS), target)
    for col, value in ((2, .5), (3, .2), (4, 1.1)):
        invalid = target.clone()
        invalid[0, 0, col] = value
        with pytest.raises(ValueError):
            action_labels(invalid)
    invalid = target.clone()
    invalid[0, 1, 4] = .7
    with pytest.raises(ValueError):
        action_labels(invalid)


def test_weights_count_actual_overlapping_chunks_and_reset_at_episode_boundary(tmp_path):
    paths = []
    counts_brute = np.zeros(5, dtype=int)
    base_brute, grip_brute = np.zeros((5, 2), dtype=int), np.zeros((2, 2), dtype=int)
    for i, target in enumerate((physical_targets()[0], physical_targets()[0, 3:])):
        state = target.clone()
        state[::2, 2:] = 0
        state[0, 4] = .8
        path = tmp_path / f'{i}.h5'
        with h5py.File(path, 'w') as f:
            f['action'] = target.numpy()
            f['observations/state'] = state.numpy()
        paths.append(path)
        for start in range(len(target)):
            chunk = target[start:start + 3]
            modes, grip, _ = action_labels(chunk)
            base_switch, grip_switch = transition_masks(chunk, state[start:start + 3])
            counts_brute += np.bincount(modes.numpy(), minlength=5)
            np.add.at(base_brute, (modes.numpy(), base_switch.long().numpy()), 1)
            np.add.at(grip_brute, (grip.numpy(), grip_switch.long().numpy()), 1)
    balance = training_class_balance(SimpleNamespace(paths=paths, cfg=SimpleNamespace(chunk_size=3)))
    assert balance['counts'] == counts_brute.tolist()
    np.testing.assert_allclose(np.array(balance['weights']) * counts_brute,
                               counts_brute.sum() / 5)
    assert balance['base_group_counts'] == base_brute.tolist()
    assert balance['gripper_group_counts'] == grip_brute.tolist()
    for counts, weights, totals in (
        (base_brute, balance['base_group_weights'], np.full(5, counts_brute.sum() / 5)),
        (grip_brute, balance['gripper_group_weights'], grip_brute.sum(1)),
    ):
        weighted = counts * np.asarray(weights)
        np.testing.assert_allclose(weighted.sum(1), totals)
        both = (counts > 0).all(1)
        np.testing.assert_allclose(weighted[both, 0], weighted[both, 1])


def test_weighted_ce_does_not_cancel_rare_weight_in_single_class_batch():
    output = heads(1)
    target = normalized(physical_targets()[:, 3:4])
    pad = torch.zeros(1, 1, dtype=torch.bool)
    state = torch.zeros_like(target)
    _, plain, _ = hybrid_loss(output, target, pad, STATS, balance(), state)
    _, weighted, _ = hybrid_loss(output, target, pad, STATS, balance([1., 1., 1., 4., 1.]), state)
    torch.testing.assert_close(weighted['base_ce'], 4 * plain['base_ce'])
    grad = torch.autograd.grad(weighted['base_ce'], output['base_logits'])[0]
    assert grad[0, 0, 3] < 0 and (grad[0, 0, [0, 1, 2, 4]] > 0).all()


def test_padding_is_excluded_and_amplitude_uses_true_not_predicted_class():
    output = heads()
    # Every prediction's argmax is stop, but true turns must still train amplitude.
    target = normalized(physical_targets())
    pad = torch.tensor([[False, False, False, False, True]])
    target[pad] = float('nan')
    state = torch.zeros_like(target)
    state[pad] = float('nan')
    for value in output.values():
        with torch.no_grad():
            value[pad] = float('nan')
    loss, _, _ = hybrid_loss(output, target, pad, STATS, balance(), state)
    assert torch.isfinite(loss)
    loss.backward()
    assert output['turn'].grad[0, 3, 0] != 0
    assert torch.all(output['turn'].grad[0, :3] == 0)
    for value in output.values():
        assert torch.isfinite(value.grad).all() and (value.grad[pad] == 0).all()
    with pytest.raises(ValueError, match='valid actions'):
        hybrid_loss(heads(), target, torch.ones_like(pad), STATS, balance(), state)


def test_validation_loss_does_not_depend_on_batching_or_no_turn_batches():
    output = heads()
    target = normalized(physical_targets())
    state = torch.zeros_like(target)
    all_loss, _, _ = hybrid_loss(output, target, torch.zeros(1, 5, dtype=torch.bool), STATS, balance(), state)
    average = LossAverages()
    for start, end in ((0, 3), (3, 5)):
        batch = {key: value[:, start:end] for key, value in output.items()}
        _, parts, counts = hybrid_loss(batch, target[:, start:end], torch.zeros(1, end-start, dtype=torch.bool),
                                      STATS, balance(), state[:, start:end])
        average.update(parts, counts)
    assert average.result('val')['val_loss'] == pytest.approx(float(all_loss.detach()))


def test_temporal_fusion_chooses_a_class_before_decoding_speed():
    left, right = heads(2), heads(2)
    left['base_logits'] = torch.tensor([[[0., 0., 0., 8., 0.]]]).expand(1, 2, 5)
    right['base_logits'] = torch.tensor([[[0., 0., 0., 0., 7.]]]).expand(1, 2, 5)
    left['turn'] = torch.full((1, 2, 1), .9)
    right['turn'] = torch.full((1, 2, 1), .6)
    ensemble = TemporalEnsembler(0)
    ensemble.select(0, pack_heads(left)[0].detach().numpy())
    merged = ensemble.select(1, pack_heads(right)[0].detach().numpy())
    action = decode_packet(torch.from_numpy(merged), STATS)
    assert action[3] == 0 and action[4] > .89  # Not the signed-speed average ~.15.


def test_temporal_fusion_clamps_raw_amplitude_only_after_averaging():
    first, second = heads(1), heads(1)
    for output, raw in ((first, -2.), (second, 1.)):
        output['base_logits'] = torch.tensor([[[0., 0., 0., 20., 0.]]])
        output['turn'] = torch.full((1, 1, 1), raw)
    merged = (pack_heads(first) + pack_heads(second)) / 2
    assert decode_packet(merged, STATS)[0, 0, 4] == 0  # clamp(mean(-2,1)), not mean(clamp(-2),clamp(1)).


@pytest.mark.parametrize('version', [None, 1, 3])
def test_checkpoint_rejects_unsupported_formats(tmp_path, version):
    from act.checkpoint import load_checkpoint, save_checkpoint
    path = tmp_path / 'unsupported.pt'
    save_checkpoint(path, {'format_version': version})
    with pytest.raises(ValueError, match='requires a hybrid ACT checkpoint'):
        load_checkpoint(path)


def test_checkpoint_rejects_changed_action_contract(tmp_path):
    from act.action_codec import CONTRACT
    from act.checkpoint import load_checkpoint, save_checkpoint
    path = tmp_path / 'wrong-codec.pt'
    save_checkpoint(path, {'format_version': 2, 'action_codec': {**CONTRACT, 'drive_speed_m_s': .5}})
    with pytest.raises(ValueError, match='action contract'):
        load_checkpoint(path)


def test_old_sigmoid_checkpoint_is_rejected(tmp_path):
    from act.action_codec import CONTRACT
    from act.checkpoint import load_checkpoint, save_checkpoint
    path = tmp_path / 'old-sigmoid.pt'
    old = {k: v for k, v in CONTRACT.items() if k not in ('turn_head', 'supervision')}
    old['version'] = 1
    save_checkpoint(path, {'format_version': 2, 'action_codec': old})
    with pytest.raises(ValueError, match='action contract'):
        load_checkpoint(path)


def test_switches_use_measured_state_and_distinguish_direction_changes():
    # Stop, start forward/back, start/reverse left/right, and ordinary motion.
    target = torch.tensor([[0., 0., 1., 0., 0.], [0., 0., 0., .3, 0.],
                           [0., 0., 1., -.3, 0.], [0., 0., 0., 0., .7],
                           [0., 0., 1., 0., -.7], [0., 0., 1., 0., .7],
                           [0., 0., 0., 0., 0.], [0., 0., 1., .3, 0.]])
    state = torch.zeros_like(target)
    state[:, 2] = torch.tensor([0., 1., 1., 0., .5, 1., 0., 1.])
    state[:, 4] = torch.tensor([.8, 0., 0., -.8, 0., .8, 0., 0.])
    state[2, 3], state[7, 3] = .3, .3
    base, grip = transition_masks(target, state)
    assert base.tolist() == [True] * 5 + [False] * 3
    assert grip.tolist() == [True, True, False, False, False, False, False, False]


def test_switch_ce_changes_weight_not_label_or_amplitude_supervision():
    target = physical_targets()[:, :1].expand(1, 2, 5).clone()  # Two stop/open targets.
    state = target.clone()
    state[0, 1, 4], state[0, 1, 2] = .8, 0  # Only the second needs stop/open switches.
    weights = balance()
    weights['base_group_weights'][0] = [.5, 4.]
    weights['gripper_group_weights'][1] = [.75, 3.]
    output = heads(2)
    loss, parts, _ = hybrid_loss(output, normalized(target), torch.zeros(1, 2, dtype=torch.bool),
                                 STATS, weights, state)
    assert parts['base_ce'].item() == pytest.approx(np.log(5) * 2.25)
    assert parts['gripper_ce'].item() == pytest.approx(np.log(2) * 1.875)
    loss.backward()
    assert output['base_logits'].grad[0, 1, 0] == pytest.approx(
        output['base_logits'].grad[0, 0, 0].item() * 8)
    assert torch.all(output['turn'].grad == 0)  # Stop labels never supervise amplitude.


def test_missing_subgroup_keeps_nonzero_validation_weight(tmp_path):
    path = tmp_path / 'ordinary.h5'
    with h5py.File(path, 'w') as f:
        f['action'] = np.zeros((2, 5), dtype=np.float32)
        f['observations/state'] = np.zeros((2, 5), dtype=np.float32)
    result = training_class_balance(SimpleNamespace(paths=[path], cfg=SimpleNamespace(chunk_size=2)))
    assert result['base_group_weights'][0] == [1., 1.]
    assert result['gripper_group_weights'] == [[1., 1.], [1., 1.]]


@pytest.mark.parametrize('raw', [-.4, 1.6])
def test_raw_amplitude_mse_has_gradients_outside_execution_limits(raw):
    target = physical_targets()[:, 3:4]
    output = heads(1)
    output['turn'] = torch.tensor([[[raw]]], requires_grad=True)
    output['base_logits'] = torch.tensor([[[0., 0., 0., 20., 0.]]])
    _, parts, _ = hybrid_loss(output, normalized(target), torch.zeros(1, 1, dtype=torch.bool),
                             STATS, balance(), torch.zeros_like(target))
    parts['turn_mse'].backward()
    assert parts['turn_mse'].item() == pytest.approx((raw - .7) ** 2)
    assert output['turn'].grad.item() == pytest.approx(2 * (raw - .7))
    assert decode_heads(output, STATS)[0, 0, 4].item() == pytest.approx(max(0., min(1., raw)))
