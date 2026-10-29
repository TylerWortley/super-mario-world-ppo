# super-mario-world-ppo

PPO written from scratch in PyTorch, trained on Super Mario World and tested on unseen levels.

My general timeline/plan for this project:

- [ ] REINFORCE on CartPole (whole-episode return vs reward-to-go)
- [ ] REINFORCE with a learned baseline
- [ ] Generalized Advantage Estimation (GAE), with unit tests
- [ ] PPO on CartPole, matched against CleanRL's reference implementation
- [ ] PPO on Super Mario World pixels (Yoshi's Island 1)
- [ ] Generalization study: train on 18 levels, test on 6 held-out levels

**Status:** in progress

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```
git clone https://github.com/TylerWortley/super-mario-world-ppo.git
cd super-mario-world-ppo
uv sync
```

Run the tests:

```
uv run pytest
```

The Super Mario World ROM is not included - your own legally obtained copy is required.

## Layout

```
ppo/       algorithms (advantages, PPO, networks)
smw/       Super Mario World environment wrappers and evaluation
scripts/   training and evaluation entry points
tests/     unit tests
results/   evaluation outputs and plots
```
