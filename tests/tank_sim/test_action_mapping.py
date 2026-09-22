"""动作映射单测。"""

import numpy as np

from tank_sim.control.action_mapping import continuous_to_intent
from tank_sim.core.types import MoveIntent, RotateIntent


def test_deadzone_stop():
    intent = continuous_to_intent(np.array([0.0, 0.0, -1.0]))
    assert intent.move == MoveIntent.STOP
    assert intent.rotate == RotateIntent.STOP
    assert intent.fire is False


def test_forward_fire():
    intent = continuous_to_intent(np.array([0.9, 0.9, 0.9]))
    assert intent.move == MoveIntent.FORWARD
    assert intent.rotate == RotateIntent.RIGHT
    assert intent.fire is True


def test_fire_threshold_positive_only():
    assert continuous_to_intent(np.array([0.0, 0.0, 0.01])).fire is True
    assert continuous_to_intent(np.array([0.0, 0.0, 0.0])).fire is False
    assert continuous_to_intent(np.array([0.0, 0.0, -0.1])).fire is False
    assert continuous_to_intent(np.array([0.0, 0.0, 0.4])).fire is True
