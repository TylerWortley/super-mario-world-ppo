"""PPO on Super Mario World implemented by hand (extension of ppo/ppo.py)

Settings are aligned with CleanRL's Atari PPO. Checkpoints saved to runs/ppo_mario/{state}_seed{seed}.pt
For every finished episode, step, return, and Mario's x position upon termination/truncation is saved to
runs/ppo_mario/{state}_seed{seed}.npz.
With --snapshot-every N, a copy of the weights is also kept every N steps in
runs/ppo_mario/{state}_seed{seed}_snapshots/step{global_step}.pt

Usage: uv run python -m ppo.ppo_mario --seed 0 --total-timesteps 10000000
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

from ppo.advantages import generalized_advantage_estimate
from smw.env import make_env

ROOT = Path(__file__).resolve().parents[1]


def initialize_layer(layer, std=2**0.5, bias=0.0):
    """Orthogonal weights and constant biases to replicate CleanRL's PPO implementation"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """DQN CNN with shared actor-critic backbone"""

    def __init__(self, n_actions):
        super().__init__()
        self.encoder = nn.Sequential(
            initialize_layer(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            initialize_layer(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            initialize_layer(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            initialize_layer(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = initialize_layer(nn.Linear(512, n_actions), std=0.01)
        self.critic = initialize_layer(nn.Linear(512, 1), std=1.0)

    def encode(self, obs):
        return self.encoder(obs.float() / 255.0)

    def get_value(self, obs):
        """Computes V(s) for a batch of observations"""
        return self.critic(self.encode(obs)).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        """Compute log prob, entropy, and value given an action.
        With action=None, sample a new action from the policy
        Otherwise score given actions under the current policy
        """

        features = self.encode(obs)
        dist = Categorical(logits=self.actor(features))
        if action is None:
            action = dist.sample()
        value = self.critic(features).squeeze(-1)

        return action, dist.log_prob(action), dist.entropy(), value


def main():
    parser = argparse.ArgumentParser(
        description="Run PPO algorithm on Super Mario World"
    )
    parser.add_argument(
        "--state", default="YoshiIsland1", help="Super Mario World level"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments for training",
    )
    parser.add_argument(
        "--num-steps", type=int, default=128, help="Number of steps per environment"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Controls how much agent cares about immediate versus future rewards",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.95,
        help="Controls short step vs long step advantage estimates (TD to MC)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Clipping hyperparameter used in surrogate objective function to constrain how much policy can change in a single update",
    )
    parser.add_argument(
        "--c1",
        type=float,
        default=0.5,
        help="Value function coefficient. Controls weight given to the value function error in loss function.",
    )
    parser.add_argument(
        "--c2",
        type=float,
        default=0.01,
        help="Entropy coefficient. Controls weight given to entropy bonus in loss function, lower is more deterministic, higher encourages more exploration",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Controls maximum L2 norm of the gradient vector across both actor and critic neural networks.",
    )
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for random generation"
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="Also keep a step-numbered checkpoint every this many steps (0 = off)",
    )

    args = parser.parse_args()

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    torch.manual_seed(args.seed)

    # create multiple sub-environments with batched inference
    envs = gym.vector.AsyncVectorEnv(
        [lambda: make_env(args.state)] * args.num_envs,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    obs_shape = envs.single_observation_space.shape
    agent = Agent(envs.single_action_space.n).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # rollout buffers, one row per step, one col per env
    shape = (args.num_steps, args.num_envs)
    obs_buf = torch.zeros((*shape, *obs_shape), dtype=torch.uint8, device=device)
    actions_buf = torch.zeros(shape, dtype=torch.int64, device=device)
    log_probs_buf = torch.zeros(shape, device=device)
    values_buf = torch.zeros(shape, device=device)
    next_values_buf = torch.zeros(shape, device=device)
    rewards_buf = torch.zeros(shape)
    terminated_buf = torch.zeros(shape)
    truncated_buf = torch.zeros(shape)

    obs, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs, device=device)
    global_step = 0
    episode_steps = []
    episode_returns = []
    episode_xs = []
    run_dir = ROOT / "runs" / "ppo_mario"
    run_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.state}_seed{args.seed}"
    snapshot_dir = run_dir / f"{name}_snapshots"
    if args.snapshot_every:
        snapshot_dir.mkdir(exist_ok=True)
    start = time.perf_counter()

    for update in range(num_updates):
        # lr ramping update
        frac = 1 - update / num_updates
        optimizer.param_groups[0]["lr"] = frac * args.lr

        update_returns = []
        update_xs = []

        # rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = obs

            with torch.no_grad():
                # select action
                action, log_prob, _, value = agent.get_action_and_value(obs)
                actions_buf[step] = action
                log_probs_buf[step] = log_prob
                values_buf[step] = value

                # take action and fill buffers
                next_obs, reward, terminated, truncated, infos = envs.step(
                    action.cpu().numpy()
                )
                next_obs = torch.as_tensor(next_obs, device=device)
                rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32)
                terminated_buf[step] = torch.as_tensor(terminated, dtype=torch.float32)
                truncated_buf[step] = torch.as_tensor(truncated, dtype=torch.float32)

            with torch.no_grad():
                next_value = agent.get_value(next_obs)

                # if environment was reset, replace new first observation bootstrapped from the old episode's final obs
                if "_final_obs" in infos and infos["_final_obs"].any():
                    ended = infos["_final_obs"]
                    final_obs = np.stack(infos["final_obs"][ended])
                    final_obs = torch.as_tensor(final_obs, device=device)
                    next_value[torch.as_tensor(ended)] = agent.get_value(final_obs)

            next_values_buf[step] = next_value

            if "_episode" in infos:
                done = infos["_episode"]
                final_xs = infos["final_info"]["x"][done]
                for episode_return, x in zip(infos["episode"]["r"][done], final_xs):
                    episode_steps.append(global_step)
                    episode_returns.append(float(episode_return))
                    episode_xs.append(int(x))
                    update_returns.append(float(episode_return))
                    update_xs.append(int(x))
            obs = next_obs

        # advantages
        advantages, returns = generalized_advantage_estimate(
            rewards_buf.numpy(),
            values_buf.cpu().numpy(),
            next_values_buf.cpu().numpy(),
            terminated_buf.numpy(),
            truncated_buf.numpy(),
            gamma=args.gamma,
            lam=args.lam,
        )

        # flatten into one batch
        b_obs = obs_buf.reshape(-1, *obs_shape)
        b_actions = actions_buf.reshape(-1)
        b_log_probs = log_probs_buf.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_advantages = torch.as_tensor(
            advantages, dtype=torch.float32, device=device
        ).reshape(-1)
        b_returns = torch.as_tensor(
            returns, dtype=torch.float32, device=device
        ).reshape(-1)

        # update
        approx_kls = []
        clip_fractions = []
        for epoch in range(args.update_epochs):
            order = torch.randperm(batch_size, device=device)
            for begin in range(0, batch_size, minibatch_size):
                mb = order[begin : begin + minibatch_size]
                _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb]
                )

                ratio = torch.exp(new_log_prob - b_log_probs[mb])

                mb_advantages = b_advantages[mb]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # clipped surrogate objective
                surrogate = ratio * mb_advantages
                clipped_surrogate = (
                    torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
                    * mb_advantages
                )
                pg_loss = -torch.min(surrogate, clipped_surrogate).mean()

                # clipped value loss replicated from CleanRL
                v_clipped = b_values[mb] + torch.clamp(
                    new_value - b_values[mb], -args.epsilon, args.epsilon
                )
                v_loss_unclipped = (new_value - b_returns[mb]) ** 2
                v_loss_clipped = (v_clipped - b_returns[mb]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = pg_loss + args.c1 * v_loss - args.c2 * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # diagnostics: approximate KL and fraction of ratios outside the clip
                with torch.no_grad():
                    approx_kls.append(((ratio - 1) - ratio.log()).mean().item())
                    clipped = (ratio - 1).abs() > args.epsilon
                    clip_fractions.append(clipped.float().mean().item())
        # logging
        var_returns = b_returns.var().item()
        explained_var = 1 - (b_returns - b_values).var().item() / var_returns

        if update % 50 == 0 or update == num_updates - 1:
            mean_return = np.mean(update_returns) if update_returns else float("nan")
            mean_x = np.mean(update_xs) if update_xs else float("nan")
            print(
                f"update {update:4d}  step {global_step:8d}  return {mean_return:6.1f}  "
                f"x {mean_x:5.0f}  kl {np.mean(approx_kls):.4f}  "
                f"clipfrac {np.mean(clip_fractions):.3f}  "
                f"entropy {entropy.mean().item():.3f}  v_loss {v_loss.item():6.1f}  "
                f"ev {explained_var:5.2f}  "
                f"sps {global_step / (time.perf_counter() - start):.0f}",
                flush=True,
            )
            torch.save(agent.state_dict(), run_dir / f"{name}.pt")
            np.savez(
                run_dir / f"{name}.npz",
                steps=np.array(episode_steps),
                returns=np.array(episode_returns),
                xs=np.array(episode_xs),
            )

        # saving doesn't use any random numbers, so the run itself is unchanged
        if args.snapshot_every and update % (args.snapshot_every // batch_size) == 0:
            torch.save(agent.state_dict(), snapshot_dir / f"step{global_step:08d}.pt")

    envs.close()
    print(f"saved {run_dir / name}.pt and .npz")


if __name__ == "__main__":
    main()
