import pytest
import torch

from act.metrics import MotionMetrics


def test_zero_policy_has_low_global_error_but_no_turn_recall():
    target = torch.zeros(1, 100, 5)
    target[0, 0, 4], target[0, 1, 4] = 1, -1
    pred = torch.zeros_like(target)
    metrics = MotionMetrics('cpu')
    metrics.update(pred, target, torch.zeros(1, 100, dtype=torch.bool), torch.zeros(1, 5))
    result = metrics.result()
    assert result['val_zero_turn_mae'] == pytest.approx(.02)
    assert result['val_turn_recall'] == 0
    assert result['val_first_turn_start_recall'] == 0
    assert result['val_turn_mae_vs_zero'] == 1


def test_wrong_direction_padding_and_close_transition_are_distinguished():
    target = torch.zeros(2, 2, 5)
    target[:, 0, 4] = torch.tensor([1., -1.])
    target[:, 1, 4] = 100  # Padded garbage must not count.
    pred = target.clone()
    pred[1, 0, 4] = 1  # Right turn predicted as left.
    pred[1, 0, 2] = 1  # Fails to close from open.
    state = torch.zeros(2, 5)
    state[:, 2] = 1
    state[1, 4] = -.5  # Already turning; only sample 0 is a start.
    metrics = MotionMetrics('cpu')
    metrics.update(pred, target, torch.tensor([[False, True], [False, True]]), state)
    result = metrics.result()
    assert result['val_turn_active_count'] == 2
    assert result['val_turn_recall'] == .5
    assert result['val_first_turn_start_recall'] == 1
    assert result['val_first_close_from_open_recall'] == .5
    assert result['val_turn_false_positive_rate'] is None


def test_continuing_to_turn_is_not_a_correct_stop():
    target = torch.zeros(2, 1, 5)
    pred = target.clone()
    pred[0, 0, 4] = -.9
    state = torch.zeros(2, 5)
    state[:, 4] = -.8
    metrics = MotionMetrics('cpu')
    metrics.update(pred, target, torch.zeros(2, 1, dtype=torch.bool), state)
    result = metrics.result()
    assert result['val_first_turn_stop_count'] == 2
    assert result['val_first_turn_stop_recall'] == .5
    assert result['val_first_turn_false_positive_rate'] == .5
    assert result['val_first_turn_stop_mae'] == pytest.approx(.45)


def test_left_and_right_starts_are_not_hidden_in_one_average():
    target = torch.zeros(2, 1, 5)
    target[:, 0, 4] = torch.tensor([1., -1.])
    pred = target.clone()
    pred[1, 0, 4] = 0
    metric = MotionMetrics('cpu')
    metric.update(pred, target, torch.zeros(2, 1, dtype=torch.bool), torch.zeros(2, 5))
    result = metric.result()
    assert result['val_first_turn_left_start_recall'] == 1
    assert result['val_first_turn_right_start_recall'] == 0


def test_adapter_does_not_apply_a_second_slew_limit_by_default():
    import numpy as np
    from act.rollout import SimActionAdapter
    class Robot:
        def __init__(self):
            self.applied_action = np.array([0., 0., 1., 0., 0.])
        def arm_geometry(self):
            return 1, 1, np.array([-2., -2.]), np.array([2., 2.])
        def command_arm(self, q):
            self.applied_action[:2] = q
        def command_drive(self, v, w):
            self.applied_action[3:] = v, w
    # Teacher ramps from measured q; 0.307 is a real recorded command increment.
    direct = SimActionAdapter(Robot())
    assert direct.apply([.307, 0, 1, 0, 0])[0] == pytest.approx(.307)


def test_replay_preserves_repeated_closed_commands_as_distinct_events():
    from act.replay import gripper_events
    rows = [{'actions': [
        {'action': 'close_gripper', 'action_applied': True, 'frame_range': [5, 17]},
        {'action': 'close_gripper', 'action_applied': True, 'frame_range': [17, 29]},
        {'action': 'close_gripper', 'action_applied': False, 'frame_range': None},
    ]}]
    starts, ends = gripper_events(rows, 40)
    assert starts == {5: 'close_gripper', 17: 'close_gripper'}
    assert ends == {17, 29}
