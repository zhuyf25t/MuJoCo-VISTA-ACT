"""Head/label units, weighted gradients, masked losses and checkpoint validation."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from act.action_codec import action_labels, decode_heads, decode_packet, pack_heads
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
    for i, target in enumerate((physical_targets()[0], physical_targets()[0, 3:])):
        path = tmp_path / f'{i}.h5'
        with h5py.File(path, 'w') as f:
            f['action'] = target.numpy()
        paths.append(path)
        for start in range(len(target)):
            modes, _, _ = action_labels(target[start:start + 3])
            counts_brute += np.bincount(modes.numpy(), minlength=5)
    balance = training_class_balance(SimpleNamespace(paths=paths, cfg=SimpleNamespace(chunk_size=3)))
    assert balance['counts'] == counts_brute.tolist()
    np.testing.assert_allclose(np.array(balance['weights']) * counts_brute,
                               counts_brute.sum() / 5)


def test_weighted_ce_does_not_cancel_rare_weight_in_single_class_batch():
    output = heads(1)
    target = normalized(physical_targets()[:, 3:4])
    pad = torch.zeros(1, 1, dtype=torch.bool)
    _, plain, _ = hybrid_loss(output, target, pad, STATS, [1.] * 5)
    _, weighted, _ = hybrid_loss(output, target, pad, STATS, [1., 1., 1., 4., 1.])
    torch.testing.assert_close(weighted['base_ce'], 4 * plain['base_ce'])
    grad = torch.autograd.grad(weighted['base_ce'], output['base_logits'])[0]
    assert grad[0, 0, 3] < 0 and (grad[0, 0, [0, 1, 2, 4]] > 0).all()


def test_padding_is_excluded_and_amplitude_uses_true_not_predicted_class():
    output = heads()
    # Every prediction's argmax is stop, but true turns must still train amplitude.
    target = normalized(physical_targets())
    pad = torch.tensor([[False, False, False, False, True]])
    target[pad] = float('nan')
    for value in output.values():
        with torch.no_grad():
            value[pad] = float('nan')
    loss, _, _ = hybrid_loss(output, target, pad, STATS, [1.] * 5)
    assert torch.isfinite(loss)
    loss.backward()
    assert output['turn'].grad[0, 3, 0] != 0
    assert torch.all(output['turn'].grad[0, :3] == 0)
    for value in output.values():
        assert torch.isfinite(value.grad).all() and (value.grad[pad] == 0).all()
    with pytest.raises(ValueError, match='valid actions'):
        hybrid_loss(heads(), target, torch.ones_like(pad), STATS, [1.] * 5)


def test_validation_loss_does_not_depend_on_batching_or_no_turn_batches():
    output = heads()
    target = normalized(physical_targets())
    all_loss, _, _ = hybrid_loss(output, target, torch.zeros(1, 5, dtype=torch.bool), STATS, [1.] * 5)
    average = LossAverages()
    for start, end in ((0, 3), (3, 5)):
        batch = {key: value[:, start:end] for key, value in output.items()}
        _, parts, counts = hybrid_loss(batch, target[:, start:end], torch.zeros(1, end-start, dtype=torch.bool), STATS, [1.] * 5)
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
