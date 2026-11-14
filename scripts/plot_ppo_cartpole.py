"""Plot my PPO against CleanRL's ppo.py on CartPole-v1, 3 seeds each

Reads runs/ppo/cartpole_seed{seed}.npz from ppo/ppo.py and
runs/cleanrl/cleanrl_seed{seed}_charts_episodic_return.csv from CleanRL's TensorBoard
logs. Each curve is the mean episode return in 10k-step bins; the band is the
min-max over seeds.

Usage: uv run python scripts/plot_ppo_cartpole.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
OUTPUT_PATH = ROOT / "results" / "ppo_cartpole_vs_cleanrl.png"

SEEDS = [0, 1, 2]
TOTAL_STEPS = 500_000
BIN = 10_000


def load_mine(seed):
    run = np.load(RUNS_DIR / "ppo" / f"cartpole_seed{seed}.npz")
    return run["steps"], run["returns"]


def load_cleanrl(seed):
    path = RUNS_DIR / "cleanrl" / f"cleanrl_seed{seed}_charts_episodic_return.csv"
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def binned(steps, returns):
    """Mean return of the episodes that finished in each BIN-step window."""
    which = np.minimum(steps // BIN, TOTAL_STEPS // BIN - 1).astype(int)
    sums = np.bincount(which, weights=returns, minlength=TOTAL_STEPS // BIN)
    counts = np.bincount(which, minlength=TOTAL_STEPS // BIN)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def summary(steps, returns):
    hit = steps[returns >= 500]
    first = int(hit[0]) if hit.size else None
    return first, returns[steps >= TOTAL_STEPS - 50_000].mean()


def main():
    fig, ax = plt.subplots(figsize=(8, 5))
    x = (np.arange(TOTAL_STEPS // BIN) + 0.5) * BIN

    methods = [("CleanRL ppo.py", load_cleanrl, "C7"), ("my PPO", load_mine, "C0")]
    for label, load, color in methods:
        runs = [load(seed) for seed in SEEDS]
        curves = np.stack([binned(steps, returns) for steps, returns in runs])
        ax.fill_between(
            x, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.2
        )
        ax.plot(x, curves.mean(axis=0), color=color, lw=2, label=label)

        for seed, (steps, returns) in zip(SEEDS, runs):
            first, last_mean = summary(steps, returns)
            print(
                f"{label:15s} seed {seed}: first 500 at step {first}, "
                f"mean return over the last 50k steps {last_mean:.1f}"
            )

    ax.axhline(500, color="gray", ls="--", lw=1, label="time limit (500 steps)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"mean episode return ({BIN // 1000}k-step bins)")
    ax.set_title(f"PPO on CartPole-v1 ({len(SEEDS)} seeds, band = min to max)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
