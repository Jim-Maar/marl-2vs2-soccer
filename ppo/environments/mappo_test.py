import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import gymnasium as gym
import numpy as np
import torch as t
from gymnasium.spaces import Box, Discrete, MultiDiscrete

warnings.filterwarnings("ignore")

Arr = np.ndarray


device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %%

ObsType: TypeAlias = int | np.ndarray
ActType: TypeAlias = int

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
NO_OP = 4

class MappoTest(gym.Env):
    """One action, observation of [0.0], one timestep long, +1 reward.

    We expect the agent to rapidly learn that the value of the constant [0.0] observation is +1.0. Note we're using a continuous observation space for consistency with CartPole.
    """

    action_space: MultiDiscrete
    observation_space: Box

    def get_observations(self) -> np.ndarray:
        x1 = self.agent_positions[0, 0]
        y1 = self.agent_positions[0, 1]
        x2 = self.agent_positions[1, 0]
        y2 = self.agent_positions[1, 1]
        return np.array([[x1, y1, x2, y2], [x2, y2, x1, y1]])

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()
        self.num_agents = 2
        self.max_steps = 10
        self.observation_space = Box(
            low=np.array([[0, 0, 0, 0], [0, 0, 0, 0]]),
            high=np.array([[2, 2, 2, 2], [2, 2, 2, 2]]),
        )
        self.action_space = MultiDiscrete([5, 5])
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        for i in range(self.num_agents):
            if action[i] == UP:
                self.agent_positions[i, 1] = min(self.agent_positions[i, 1] + 1, 2)
            elif action[i] == DOWN:
                self.agent_positions[i, 1] = max(self.agent_positions[i, 1] - 1, 0)
            elif action[i] == LEFT:
                self.agent_positions[i, 0] = max(self.agent_positions[i, 0] - 1, 0)
            elif action[i] == RIGHT:
                self.agent_positions[i, 0] = min(self.agent_positions[i, 0] + 1, 2)
            elif action[i] == NO_OP:
                pass
        reward = 1.0 if np.array_equal(self.agent_positions[0], self.agent_positions[1]) else 0.0
        truncated = self.num_steps >= self.max_steps
        terminated = False
        return self.get_observations(), reward, terminated, truncated, {}

    def reset(self, seed: int | None = None, options=None) -> ObsType | tuple[ObsType, dict]:
        super().reset(seed=seed)
        self.agent_positions = np.array([[0, 0], [0, 2]])
        self.num_steps = 0
        return self.get_observations(), {}

if MAIN:
    gym.envs.registration.register(id="MappoTest-v0", entry_point=MappoTest)
    env = gym.make("MappoTest-v0")
    assert env.observation_space.shape == (2, 4)
    assert env.action_space.shape == (2,)