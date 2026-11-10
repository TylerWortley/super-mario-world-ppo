"""Return and advantage estimators for policy-gradient methods."""

import numpy as np


def rewards_to_go(rewards, gamma):
    """Compute the discounted reward-to-go for every step of an episode.

        G_t = r_t + gamma * r_{t+1} + gamma**2 * r_{t+2} + ... = r_t + gamma * G_{t+1}

    Args:
        rewards: Rewards for one complete episode, in time order
            (list or 1-D array).
        gamma: Discount factor in [0, 1].

    Returns:
        A new float array the same length as ``rewards``.
    """

    Gt = np.zeros(len(rewards), dtype=float)
    running_return = 0.0

    # loop from end to start rewards, to calculate reward-to-go
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        Gt[t] = running_return

    return Gt


def generalized_advantage_estimate(
    rewards, values, next_values, terminated, truncated, gamma, lam
):
    """Compute Generalized Advantage Estimate advantages and value targets for a rollout of parallel environments.

    delta_t = r_t + gamma * (1 - terminated_t) * V(s_{t+1}) - V(s_t)
    A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}, where done_t = terminated or truncated

    Args:
        rewards: Rewards r_t, shape [T, N].
        values: Critic values V(s_t), shape [T, N].
        next_values: Critic values of the next state V(s_t+1) in an episode, shape [T, N], if step t ends
                     the episode, its value comes from the episode's final obs, not next episode's first.
        terminated: Whether episode actually ended after step t.
        truncated: whether episode was cut off after step t.
        gamma: Discount factor in [0, 1].
        lam: GAE lambda in [0, 1] (0 = one-step TD, 1 = MC return with critic baseline subtracted, bootstrapped where it was cut off)

    Returns:
        advantages, returns: two new float arrays shaped [T, N], where
        critic's regression targets are: returns = advantages + values.

    """

    # convert to np arrays for when we do element wise operations
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    next_values = np.asarray(next_values, dtype=float)
    terminated = np.asarray(terminated, dtype=float)
    truncated = np.asarray(truncated, dtype=float)

    # compute TD error for all steps
    deltas = rewards + gamma * (1 - terminated) * next_values - values

    # compute GAE recursively from end to beginning
    done = np.logical_or(terminated, truncated)
    advantages = np.zeros(rewards.shape)
    running_advantage = 0.0  # used to recursively compute advantages

    for t in reversed(range(len(rewards))):
        running_advantage = deltas[t] + gamma * lam * (1 - done[t]) * running_advantage
        advantages[t] = running_advantage

    # compute return targets
    returns = advantages + values

    return advantages, returns
