"""Plot REINFORCE whole episode returns versus reward-to-go learning curves on CartPole-v1

Reads runs/reinforce/{weight}_seed{seed}.npy from reinforce_cartpole.py

Usage: uv run python scripts/plot_reinforce.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs" / "reinforce"
OUTPUT_PATH = ROOT / "results" / "reinforce_cartpole.png"

WEIGHTS = [("return", "whole-episode return", "C0"), ("rtg", "reward-to-go", "C1")]
SEEDS = [0, 1, 2]


def main():
    fig, ax = plt.subplots(figsize=(8, 5))

    for weight, label, color in WEIGHTS:
        run_data = np.stack(
            [np.load(RUNS_DIR / f"{weight}_seed{seed}.npy") for seed in SEEDS]
        )

        for curve in run_data:
            ax.plot(curve, color=color, lw=1, alpha=0.4)

        ax.plot(run_data.mean(axis=0), color=color, lw=2.5, label=label)
    ax.axhline(500, color="gray", ls="--", label="time limit (500 steps)")
    ax.set_xlabel("update (5000 env steps each)")
    ax.set_ylabel("avg episode return")
    ax.set_title(f"REINFORCE on CartPole-v1 ({len(SEEDS)} seeds, thick line = mean)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.4)
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
