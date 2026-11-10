"""Tests for ppo.advantages.

Usage: uv run pytest
"""

import numpy as np

from ppo.advantages import generalized_advantage_estimate, rewards_to_go


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


def test_gae_no_bootstrap():
    # episode that has completed should not bootstrap
    advantages, returns = generalized_advantage_estimate(
        rewards=[1, 1, 1, 1],
        values=[0.5, 0.5, 0.5, 0.5],
        next_values=[0.5, 99, 0.5, 2.0],
        terminated=[0, 1, 0, 0],
        truncated=[0, 0, 0, 0],
        gamma=1.0,
        lam=0.5,
    )

    np.testing.assert_allclose(advantages, [1.25, 0.5, 2.25, 2.5])
    np.testing.assert_allclose(returns, [1.75, 1.0, 2.75, 3.0])


def test_gae_bootstrap():
    # truncated episode bootstraps, but its advantage does not carry back
    advantages, _ = generalized_advantage_estimate(
        rewards=[1, 1, 1, 1],
        values=[0.5, 0.5, 0.5, 0.5],
        next_values=[0.5, 3.0, 0.5, 2.0],
        terminated=[0, 0, 0, 0],
        truncated=[0, 1, 0, 0],
        gamma=1.0,
        lam=0.5,
    )
    np.testing.assert_allclose(advantages, [2.75, 3.5, 2.25, 2.5])


def test_gae_two_envs():
    # testing truncated/terminated cases but with two parallel envs.
    advantages, _ = generalized_advantage_estimate(
        rewards=np.column_stack([[1, 1, 1, 1], [1, 1, 1, 1]]),
        values=np.column_stack([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]),
        next_values=np.column_stack([[0.5, 99, 0.5, 2.0], [0.5, 3.0, 0.5, 2.0]]),
        terminated=np.column_stack([[0, 1, 0, 0], [0, 0, 0, 0]]),
        truncated=np.column_stack([[0, 0, 0, 0], [0, 1, 0, 0]]),
        gamma=1.0,
        lam=0.5,
    )
    expected = np.column_stack([[1.25, 0.5, 2.25, 2.5], [2.75, 3.5, 2.25, 2.5]])
    np.testing.assert_allclose(advantages, expected)
