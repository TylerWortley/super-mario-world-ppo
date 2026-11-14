"""Super Mario World as a Gymnasium env

make_env() stacks, from emulator outwards
    stable-retro (RGB 224x256, 12 buttons, info from mario_data.json)
    Moves: Discrete(7) instead of 12 buttons
    SkipFrames: each move lasts 4 frames, with sticky actions
    MarioReward: your reward and episode endings (smw/reward.py)
    grayscale, resize to 84x84, stack the last 4 -> obs (4, 84, 84) uint8

stable-retro allows one emulator per process, so run parallel envs with
gymnasium's AsyncVectorEnv, never SyncVectorEnv.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_retro

from smw.reward import MarioReward

GAME = "SuperMarioWorld-Snes-v0"
DATA = Path(__file__).resolve().parent / "mario_data.json"

# Y = run, B = jump (hold for height), A = spin jump
MOVES = [
    ["RIGHT"],
    ["RIGHT", "Y"],
    ["RIGHT", "B"],
    ["RIGHT", "Y", "B"],
    ["RIGHT", "A"],
    ["LEFT"],
    ["B"],
]


class Moves(gym.ActionWrapper):
    """Replace 12 button controller with discrete choice in MOVES"""

    def __init__(self, env, moves=MOVES):
        super().__init__(env)
        buttons = env.unwrapped.buttons
        self.arrays = []
        for move in moves:
            pressed = np.zeros(len(buttons), dtype=np.int8)
            for button in move:
                pressed[buttons.index(button)] = 1
            self.arrays.append(pressed)
        self.action_space = gym.spaces.Discrete(len(moves))

    def action(self, action):
        return self.arrays[int(action)].copy()


class SkipFrames(gym.Wrapper):
    """Hold each move for `skip` emulator frames and sum their rewards.

    Sticky actions (Machado et al., 2018) add stochasticity to deterministic games, gives a chance
    at any time step that the env repeats the previous action instead of executing the newly chosen one
    """

    def __init__(self, env, skip=4, repeat_prob=0.25, recorder=None):
        super().__init__(env)
        self.skip = skip
        self.repeat_prob = repeat_prob
        self.recorder = recorder
        self.rng = np.random.default_rng()
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.last_action = None
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            if self.last_action is not None and self.rng.random() < self.repeat_prob:
                action = self.last_action

            self.last_action = action
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if self.recorder is not None:
                self.recorder.append((obs, self.env.unwrapped.em.get_audio()))

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


def make_env(state="YoshiIsland1", repeat_prob=0.25, recorder=None, reward=True):
    """One Super Mario World env with observations of shape (4, 84, 84) uint8

    reward = False leaves out MarioReward
    """

    env = stable_retro.make(GAME, state=state, info=str(DATA), render_mode="rgb_array")
    env = Moves(env)
    env = SkipFrames(env, skip=4, repeat_prob=repeat_prob, recorder=recorder)

    if reward:
        env = MarioReward(env)

    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    return gym.wrappers.FrameStackObservation(env, 4)
