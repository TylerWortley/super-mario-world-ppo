"""Smoke test for Super Mario World ROM in stable-retro

Plays one level pressing random buttons, printing what the environment exposes and reporting emulator fps on one core.
Optionally saves the first 10s as a video.

Usage: uv run python scripts/rom_smoke_test.py
       uv run python scripts/rom_smoke_test.py --video videos/random_yi1.mp4
"""

import argparse
import time
from pathlib import Path

import imageio
import stable_retro

GAME = "SuperMarioWorld-Snes-v0"
TOTAL_FRAMES = 600  # at 60 fps = 10s


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for SMW and measure speed performance"
    )
    parser.add_argument("--game", default=GAME, help="stable-retro game ID")
    parser.add_argument(
        "--state", default="YoshiIsland1", help="savestate to start from"
    )
    parser.add_argument("--steps", default=5000, type=int, help="random steps to time")
    parser.add_argument("--video", default=None, help="save first 600 frames to view")
    args = parser.parse_args()

    env = stable_retro.make(game=args.game, state=args.state, render_mode="rgb_array")
    obs, info = env.reset(seed=0)

    print(f"buttons: {env.buttons}")
    print(f"obs: {obs.shape}, {obs.dtype}")
    print(f"info: {info}")

    frames = []
    episodes = 0
    start = time.perf_counter()

    # full 'steps' sample, terminated/truncated resets environment
    for t in range(args.steps):
        obs, _, terminated, truncated, info = env.step(env.action_space.sample())
        if args.video and t < TOTAL_FRAMES:
            frames.append(obs)
        if terminated or truncated:
            episodes += 1
            obs, info = env.reset()

    elapsed = time.perf_counter() - start
    env.close()

    print(
        f"{args.steps / elapsed:.0f} emulator frames/s on one core "
        f"({args.steps} steps in {elapsed:.1f} s, {episodes} episodes ended)"
    )

    if args.video:
        path = Path(args.video)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, frames, fps=60)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
