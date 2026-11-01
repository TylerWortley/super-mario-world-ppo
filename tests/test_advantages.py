"""Tests for ppo.advantages.

Usage: uv run pytest
"""

import numpy as np

from ppo.advantages import rewards_to_go


def test_no_discounting():
    result = rewards_to_go([1, 1, 1], gamma=1.0)
    np.testing.assert_allclose(result, [3, 2, 1])


def test_discounting():
    result = rewards_to_go([1, 1, 1], gamma=0.5)
    np.testing.assert_allclose(result, [1.75, 1.5, 1])


def test_late_rewards():
    result = rewards_to_go([0, 0, 10], gamma=0.9)
    np.testing.assert_allclose(result, [8.1, 9, 10])


def test_input_not_modified():
    # any in-place change would corrupt the episode rewards logged later
    rewards = np.array([1.0, 2.0, 3.0])
    original = rewards.copy()
    _ = rewards_to_go(rewards, 0.9)
    np.testing.assert_array_equal(rewards, original)
