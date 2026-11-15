"""Train Stable-Baselines3's PPO on same env and settings as ppo/ppo_mario.py

Cross check for PPO by hand with the same parameters, except SB3 doesn't clip the value loss. Saves model to runs/sb3_mario/{state}_seed{seed}.zip
and one monitor CSV per env to runs/sb3_mario/{state}_seed{seed}/.

Usage: uv run python scripts/train_sb3.py --seed 0 --total-timesteps 10000000
"""

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from smw.env import make_env

ROOT = Path(__file__).resolve().parents[1]


def monitored_env(state, log_path):
    return lambda: Monitor(make_env(state), str(log_path), info_keywords=("x",))


def main():
    parser = argparse.ArgumentParser(description="Train SB3 PPO on Super Mario World")
    parser.add_argument("--state", default="YoshiIsland1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    args = parser.parse_args()

    run_dir = ROOT / "runs" / "sb3_mario"
    log_dir = run_dir / f"{args.state}_seed{args.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    envs = SubprocVecEnv(
        [monitored_env(args.state, log_dir / str(i)) for i in range(args.num_envs)]
    )

    model = PPO(
        "CnnPolicy",
        envs,
        learning_rate=lambda progress_left: 2.5e-4 * progress_left,  # linear to 0
        n_steps=128,
        batch_size=128 * args.num_envs // 4,  # 4 minibatches
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        verbose=1,
    )
    model.learn(total_timesteps=args.total_timesteps)
    model.save(run_dir / f"{args.state}_seed{args.seed}.zip")
    envs.close()
    print(f"saved {run_dir / args.state}_seed{args.seed}.zip")


if __name__ == "__main__":
    main()
