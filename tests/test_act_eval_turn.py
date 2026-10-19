import numpy as np
import pytest

from act.eval_turn import correct_decision, select_frames, summarize


def test_turn_start_continue_stop_are_distinct():
    state, action = np.zeros((6, 5)), np.zeros((6, 5))
    state[:, 4] = [0, 0, .8, -.8, .8, 0]
    action[:, 4] = [1, -1, 1, -1, 0, 0]
    expected = ['start_left', 'start_right', 'continue_left', 'continue_right', 'stop', 'nonturn']
    assert select_frames(state, action, 0) == list(enumerate(expected))
    assert len(select_frames(np.tile(state, (3, 1)), np.tile(action, (3, 1)), 1)) == 6


def test_zero_and_always_right_fail_different_tests():
    assert not correct_decision('start_left', 0)
    assert not correct_decision('start_right', 0)
    assert correct_decision('stop', 0)
    assert correct_decision('start_right', -.8)
    assert not correct_decision('stop', -.8)
    assert not correct_decision('start_left', .01)  # Correct sign but ineffective amplitude.
    rows = [dict(split='val', group='start_right', predicted_w=-.8, target_w=-1., correct=True)]
    scores = summarize(rows)
    assert scores['val']['start_right']['correct_rate'] == 1
    assert scores['val']['start_right']['mae'] == pytest.approx(.2)
    assert scores['val']['stop']['correct_rate'] is None
