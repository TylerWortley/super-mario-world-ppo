"""Custom Super Mario World reward function and end of episode tracking

MarioReward replaces stable-retro's reward defaults. On every step, it reads Mario's state:

    x, y   Mario's position in the level, in pixels (x starts at 16 on YI1)
    anim   9 from the first frame of a death, about 190 frames before lives drops
    mode   20 while Mario is inside a level
    goal   non-zero once Mario crosses the goal tape
    lives, coins, score
"""

import gymnasium as gym


class MarioReward(gym.Wrapper):
    """Reward progress throughout the level and episode ends (goal or death)

    One step corresponds to one agent step (4 emulator frames)
    """

    def __init__(self, env, max_steps=4000, patience=250):
        super().__init__(env)
        self.max_steps = max_steps  # 4000 steps is approx 4.5 min
        self.patience = patience  # max steps without reaching new max x value

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.steps = 0
        self.steps_since_progress = 0
        self.best_x = None
        return obs, info

    def step(self, action):
        obs, _, game_over, _, info = self.env.step(action)
        self.steps += 1
        x = info["x"]
        if self.best_x is None:
            self.best_x = x

        progress = max(x - self.best_x, 0)
        if progress > 0:
            self.best_x = x
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1

        died = info["anim"] == 9
        cleared = info["goal"] > 0
        reward = progress / 10 - 5 * died + 50 * cleared

        terminated = died or cleared or game_over
        truncated = (
            self.steps_since_progress >= self.patience or self.steps >= self.max_steps
        )

        return obs, reward, terminated, truncated, info
