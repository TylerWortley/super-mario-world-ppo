"""REINFORCE on CartPole-v1: whole-episode returns vs reward-to-go

Usage: uv run python scripts/reinforce_cartpole.py --weight rtg --seed 0
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

from ppo.advantages import rewards_to_go

ROOT = Path(__file__).resolve().parents[1]


def make_policy(states, hidden, actions):
    """Build MLP consisting of two tanh hidden layers that maps observations to one logit per action.

    No softmax - compute Categorical(logits=...) outside of the network to handle log-probabilities from logits
    in a numerically stable way, whereas log(softmax(x)) can result in -inf with near zero probabilities
    """

    policy = nn.Sequential(
        nn.Linear(states, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, actions),
    )
    return policy


def select_action(policy, obs) -> int:
    """Sample one action from the policy for a single observation"""
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32)  # convert obs to tensor
        logits = policy(obs_t)  # obtain nn outputs
        dist = Categorical(logits=logits)  # convert outputs to probabilities
        action = dist.sample().item()  # sample based on probabilities
        return action


def collect_batch(env, policy, batch_size, weight):
    """Play entire episodes until at least batch_size observations have been collected

    Args:
        env: an already seeded Gymnasium environment
        policy: policy neural network used to select an action
        batch_size: minimum number of steps - last episode always runs to the end
        weight: "rtg" credits each step with its reward-to-go; "return" credits with its episode's total return

    Returns:
        obs (N, obs_dim) float32, actions (N,) int64, returns (N,) float32, and a list of each finished episode's total return
    """
    batch_obs, batch_actions, batch_returns = [], [], []
    episode_returns = []

    while len(batch_obs) < batch_size:
        state, _ = env.reset()
        episode_rewards = []
        done = False

        while not done:
            # append state before taking action + getting reward
            batch_obs.append(state)

            # select action given the state and append to list of all actions taken
            action = select_action(policy, state)
            batch_actions.append(action)

            # get reward, new state, and if episode has completed given the selected action in the current state
            state, reward, terminated, truncated, _ = env.step(action)
            episode_rewards.append(reward)
            done = terminated or truncated

        # calculate returns for each step of the current episode
        if weight == "rtg":
            returns = rewards_to_go(episode_rewards, gamma=1.0)
        else:
            returns = np.full(len(episode_rewards), sum(episode_rewards))

        batch_returns.extend(returns)
        episode_returns.append(sum(episode_rewards))

    return (
        torch.tensor(np.array(batch_obs), dtype=torch.float32),
        torch.tensor(np.array(batch_actions), dtype=torch.int64),
        torch.tensor(np.array(batch_returns), dtype=torch.float32),
        episode_returns,
    )


def play_episode(env, policy) -> float:
    """Watch an episode play out under a given policy"""
    state, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        action = select_action(policy, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        total_reward += reward

    return total_reward


def main():
    parser = argparse.ArgumentParser(description="Run REINFORCE algorithm on cartpole")
    parser.add_argument(
        "--weight",
        required=True,
        choices=["return", "rtg"],
        help="What type of accumulated future reward to track, either over the entire trajectory (return) or from the current timestep onward (reward to go)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random generation"
    )
    parser.add_argument(
        "--updates", type=int, default=100, help="Number of updates REINFORCE makes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of steps collected before single update is performed",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Learning rate for updating parameters"
    )
    parser.add_argument(
        "--watch-freq",
        type=int,
        default=0,
        help="Render one episode every N updates (0=off)",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    env = gym.make("CartPole-v1")

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = make_policy(obs_dim, 64, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    env.reset(seed=args.seed)

    # create second env to watch episode play out for a specific policy
    if args.watch_freq > 0:
        watch_env = gym.make("CartPole-v1", render_mode="human")

    reward_history = []

    # training loop
    for update in range(args.updates):
        # rollout collection
        batch_obs, batch_actions, batch_returns, episode_returns = collect_batch(
            env, policy, args.batch_size, args.weight
        )

        # log-probabilities with gradients tracked
        logits = policy(batch_obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch_actions)

        # loss: (-log-prob * return)
        loss = (-log_probs * batch_returns).mean()

        # clear old gradients, compute backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        mean_return = np.mean(episode_returns)
        reward_history.append(mean_return)
        entropy = dist.entropy().mean().item()
        print(f"update {update:3d}  return {mean_return:6.1f}   entropy {entropy:.3f}")

        # watch current policy
        if args.watch_freq > 0 and update % args.watch_freq == 0:
            total_reward = play_episode(watch_env, policy)
            print(f"  watched: {total_reward:.0f}")

    # close pygame window
    if args.watch_freq > 0:
        watch_env.close()

    # save
    output_dir = ROOT / "runs" / "reinforce"
    file_path = output_dir / f"{args.weight}_seed{args.seed}.npy"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(file_path, reward_history)


if __name__ == "__main__":
    main()
