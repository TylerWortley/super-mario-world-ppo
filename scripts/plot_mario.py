"""Plot PPO learning on Yoshi's Island 1 and print evaluation table

Reads runs/ppo_mario/*.npz, runs/sb3_mario/YoshiIsland1_seed0/*.monitor.csv and
results/eval/*.csv. Saves results/yi1_training.png and prints a markdown table.

Usage: uv run python scripts/plot_mario.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
EVAL_DIR = ROOT / "results" / "eval"
OUTPUT_PATH = ROOT / "results" / "yi1_training.png"

GOAL_X = 4822
NUM_ENVS = 8
BINS = 40

TABLE = [
    ("random", "random", "-"),
    ("scripted: run and jump", "scripted", "-"),
    ("PPO by hand, seed 0", "ppo_seed0", "10M"),
    ("PPO by hand, seed 2", "ppo_seed2", "10M"),
    ("SB3 PPO, seed 0", "sb3_seed0", "10M"),
    ("PPO by hand, reward / 10, seed 1", "ppo_reward10_seed1", "2M"),
    ("PPO by hand, reward / 100, seed 1", "ppo_reward100_seed1", "2M"),
]


def load_ppo(name):
    run = np.load(RUNS_DIR / "ppo_mario" / f"YoshiIsland1_{name}.npz")
    return run["steps"], run["xs"]


def load_sb3(seed):
    """Episodes from the 8 monitor CSVs"""
    steps, xs = [], []
    log_dir = RUNS_DIR / "sb3_mario" / f"YoshiIsland1_seed{seed}"

    for path in sorted(log_dir.glob("*.monitor.csv")):
        # skip monitor's JSON line and the header
        data = np.loadtxt(path, delimiter=",", skiprows=2, ndmin=2)

        # the envs step together
        steps.append(np.cumsum(data[:, 1]) * NUM_ENVS)
        xs.append(data[:, 3])

    return np.concatenate(steps), np.concatenate(xs)


def binned_progress(steps, xs, total):
    """Mean % of the level reached by the episodes that ended in each bin."""
    progress = 100 * np.minimum(xs / GOAL_X, 1)
    which = np.minimum(steps * BINS // total, BINS - 1).astype(int)
    sums = np.bincount(which, weights=progress, minlength=BINS)
    counts = np.bincount(which, minlength=BINS)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def read_eval(name):
    with open(EVAL_DIR / f"YoshiIsland1_{name}.csv") as f:
        rows = list(csv.DictReader(f))
    cleared = sum(row["cleared"] == "True" for row in rows)
    progress = 100 * np.array([float(row["progress"]) for row in rows])
    return cleared, len(rows), progress


def main():
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5))

    total = 10_000_000
    steps_axis = (np.arange(BINS) + 0.5) * total / BINS
    runs = [
        ("PPO by hand, seed 0", load_ppo("seed0"), "C0"),
        ("PPO by hand, seed 2", load_ppo("seed2"), "C2"),
        ("SB3 PPO, seed 0", load_sb3(0), "C7"),
    ]

    for label, (steps, xs), color in runs:
        curve = binned_progress(steps, xs, total)
        left.plot(steps_axis, curve, color=color, lw=2, label=label)

    for name, color in (("scripted", "C1"), ("random", "C3")):
        _, _, progress = read_eval(name)
        left.axhline(progress.mean(), color=color, ls="--", lw=1, label=name)

    left.set_title("Yoshi's Island 1, 10M steps")
    left.set_ylabel(f"% of level reached ({total // BINS // 1000}k-step bins)")

    total = 2_000_000
    steps_axis = (np.arange(BINS) + 0.5) * total / BINS

    for label, name, color in (
        ("progress / 10", "reward10_seed1", "C3"),
        ("progress / 100", "reward100_seed1", "C0"),
    ):
        steps, xs = load_ppo(name)
        curve = binned_progress(steps, xs, total)
        right.plot(steps_axis, curve, color=color, lw=2, label=label)

    right.set_title("Reward scale, PPO by hand, seed 1, 2M steps")
    right.set_ylabel(f"% of level reached ({total // BINS // 1000}k-step bins)")

    for ax in (left, right):
        ax.xaxis.set_major_formatter(lambda value, _: f"{value / 1e6:g}M")
        ax.set_xlabel("environment steps")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)

    left.legend(loc="center right")
    right.legend(loc="lower right")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved {OUTPUT_PATH}\n")

    print("| agent | training steps | cleared | progress (% of level) |")
    print("|---|---|---|---|")
    for label, name, trained in TABLE:
        cleared, episodes, progress = read_eval(name)
        print(
            f"| {label} | {trained} | {cleared}/{episodes} | "
            f"{progress.mean():.1f} ± {progress.std():.1f} |"
        )


if __name__ == "__main__":
    main()
