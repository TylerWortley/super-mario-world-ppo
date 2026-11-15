"""Evaluate an agent on one Super Mario World level

Every agent gets the same fixed episode endings, independent of training
reward: death, goal tape, or MAX_STEPS. Sticky actions add stochasticity
to the deterministic environment.

One row per episode is written to results/eval/{state}_{agent}.csv

Usage:
    uv run python scripts/evaluate.py --agent random
    uv run python scripts/evaluate.py --agent scripted --video videos/scripted_yi1.mp4
    uv run python scripts/evaluate.py --agent ppo --checkpoint runs/ppo_mario/YoshiIsland1_seed0.pt --video videos/ppo_yi1.mp4
"""

import argparse
import csv
import subprocess
import wave
from pathlib import Path

import imageio
import imageio_ffmpeg
import numpy as np
import torch

from smw.env import MOVES, make_env

ROOT = Path(__file__).resolve().parents[1]
GOAL_X = {"YoshiIsland1": 4822}  # px value of level's goal
MAX_STEPS = 4000
SNES_FPS = 60.0988


class RandomAgent:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return int(self.rng.integers(len(MOVES)))


class ScriptedAgent:
    """Run right and keep jumping: run+jump for 12 steps, release for 4."""

    RUN, RUN_JUMP = MOVES.index(["RIGHT", "Y"]), MOVES.index(["RIGHT", "Y", "B"])

    def __init__(self, seed):
        self.t = 0

    def act(self, obs):
        self.t += 1
        return self.RUN_JUMP if self.t % 16 < 12 else self.RUN


class PPOAgent:
    """PPO by hand from ppo/ppo_mario.py, moves sampled from its policy"""

    def __init__(self, seed, checkpoint):
        from ppo.ppo_mario import Agent

        torch.manual_seed(seed)
        self.net = Agent(len(MOVES))
        self.net.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    def act(self, obs):
        with torch.no_grad():
            action, *_ = self.net.get_action_and_value(torch.as_tensor(obs)[None])
        return int(action)


class SB3Agent:
    """Stable-Baselines3's PPO from scripts/train_sb3.py"""

    def __init__(self, seed, checkpoint):
        from stable_baselines3 import PPO

        self.model = PPO.load(checkpoint, device="cpu")
        self.model.set_random_seed(seed)

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=False)
        return int(action)


AGENTS = {
    "random": RandomAgent,
    "scripted": ScriptedAgent,
    "ppo": PPOAgent,
    "sb3": SB3Agent,
}


def run_episode(agent, state, seed, recorder=None):
    env = make_env(state, recorder=recorder, reward=False)
    obs, _ = env.reset(seed=seed)
    max_x, cleared, reason = 0, False, "max steps"

    for step in range(1, MAX_STEPS + 1):
        obs, _, game_over, _, info = env.step(agent.act(obs))
        max_x = max(max_x, info["x"])
        if info["goal"] > 0:
            cleared, reason = True, "goal"
            break
        if info["anim"] == 9 or game_over:
            reason = "death"
            break
    env.close()
    return {"cleared": cleared, "max_x": max_x, "steps": step, "end": reason}


def save_video(recorder, path):
    """Write recorded frames and sound to one mp4"""
    path.parent.mkdir(parents=True, exist_ok=True)
    silent, sound = path.with_suffix(".silent.mp4"), path.with_suffix(".wav")
    imageio.mimsave(silent, [frame for frame, _ in recorder], fps=SNES_FPS)
    with wave.open(str(sound), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)  # 16-bit samples
        w.setframerate(32040)  # the SNES audio rate
        w.writeframes(np.concatenate([audio for _, audio in recorder]).tobytes())
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", silent, "-i", sound]
        + ["-c:v", "copy", "-c:a", "aac", "-shortest", path],
        check=True,
    )
    silent.unlink()
    sound.unlink()


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on one level")
    parser.add_argument("--agent", required=True, choices=list(AGENTS))
    parser.add_argument("--state", default="YoshiIsland1", help="level savestate")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", help="model file, for --agent ppo or sb3")
    parser.add_argument("--video", default=None, help="save episode 1 here (.mp4)")

    args = parser.parse_args()

    goal_x = GOAL_X[args.state]
    rows = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        recorder = [] if args.video and episode == 0 else None
        extra = [args.checkpoint] if args.agent in ("ppo", "sb3") else []
        agent = AGENTS[args.agent](seed, *extra)
        result = run_episode(agent, args.state, seed, recorder)
        result["progress"] = min(result["max_x"] / goal_x, 1.0)
        rows.append({"agent": args.agent, "episode": episode, "seed": seed, **result})
        print(
            f"episode {episode:2d}: {result['end']:9s} max x {result['max_x']:5d} "
            f"({100 * result['progress']:5.1f}%) in {result['steps']} steps"
        )
        if recorder:
            save_video(recorder, Path(args.video))

    out_path = ROOT / "results" / "eval" / f"{args.state}_{args.agent}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cleared = sum(row["cleared"] for row in rows)
    progress = 100 * np.array([row["progress"] for row in rows])
    print(
        f"{args.agent}: cleared {cleared}/{len(rows)}, progress "
        f"{progress.mean():.1f}% ± {progress.std():.1f}% of the level"
    )
    print(f"saved {out_path}")
    if args.video:
        print(f"saved {args.video}")


if __name__ == "__main__":
    main()
