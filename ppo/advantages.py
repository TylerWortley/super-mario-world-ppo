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
