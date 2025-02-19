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

    def get_local_position(self, pos: np.ndarray, index: int) -> np.ndarray:
        """
        pos: [x, y] a global position
        """
        if index == 0: # agent 1: bottum-left
            return pos
        elif index == 1: # agent 2: bottom-right
            return np.array([2 - pos[0], pos[1]])
        elif index == 2: # agent 3: top-left
            return np.array([pos[0], 2 - pos[1]])
        elif index == 3: # agent 4: top-right
            return np.array([2 - pos[0], 2 - pos[1]])

    def update_positions_global(self):
        self.agent_positions_global = self.agent_positions_local.copy()
        # agents 2 and 4 are mirrored in the x-axis (index + 1)
        # agents 3 and 4 are mirrored in the y-axis (index + 1)
        self.agent_positions_global[1, 0] = 2 - self.agent_positions_local[1, 0]
        self.agent_positions_global[2, 1] = 2 - self.agent_positions_local[2, 1]
        self.agent_positions_global[3, 0] = 2 - self.agent_positions_local[3, 0]
        self.agent_positions_global[3, 1] = 2 - self.agent_positions_local[3, 1]

    def get_observations(self) -> np.ndarray:
        # This returns the observations for each agent again
        # Then in the agent step function the value is calculated once for each team (e.g (a1, a2, a3, a4), (a3, a4, a1, a2))
        # The step function here also returns 2 rewards
        # We need to compute advantage and everything for all teams ... And handle the one team case ...
        observations = []
        for i in range(self.num_agents):
            agent_observation = []
            team_index = i // self.team_size
            team_start = team_index * self.team_size
            team_end = (team_index + 1) * self.team_size
            teammate_indeces = list(range(team_start, i)) + list(range(i + 1, team_end))
            enemy_indeces = list(range(0, team_start)) + list(range(team_end, self.num_agents))
            agent_observation.append(self.agent_positions_local[i])
            for j in teammate_indeces + enemy_indeces: # First teammates than enemies
                agent_position = self.get_local_position(self.agent_positions_global[j], i)
                agent_observation.append(agent_position)
            observations.append(np.concatenate(agent_observation))
        return np.array(observations)



    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()
        self.num_agents = 4
        self.team_size = 2
        self.num_teams = self.num_agents // self.team_size
        self.max_steps = 10
        self.observation_space = Box(
            low=np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            high=np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]),
        )
        self.action_space = MultiDiscrete([5, 5, 5, 5])
        self.reset()

    def step(self, actions: ActType) -> tuple[ObsType, float, bool, dict]:
        for i in range(self.num_agents):
            if actions[i] == UP:
                self.agent_positions_local[i, 1] = min(self.agent_positions_local[i, 1] + 1, 2)
            elif actions[i] == DOWN:
                self.agent_positions_local[i, 1] = max(self.agent_positions_local[i, 1] - 1, 0)
            elif actions[i] == LEFT:
                self.agent_positions_local[i, 0] = max(self.agent_positions_local[i, 0] - 1, 0)
            elif actions[i] == RIGHT:
                self.agent_positions_local[i, 0] = min(self.agent_positions_local[i, 0] + 1, 2)
            elif actions[i] == NO_OP:
                pass

        self.update_positions_global()
        rewards = np.zeros(self.tean_size)
        for i in range(self.team_size):
            if np.array_equal(self.agent_positions_global[0], self.agent_positions_global[1]):
                rewards[i] = 1.0
        truncated = False
        terminated = False
        self.num_steps += 1
        if self.num_steps >= self.max_steps:
            truncated = True
            self.reset()
        return self.get_observations(), rewards, terminated, truncated, {}

    def reset(self, seed: int | None = None, options=None) -> ObsType | tuple[ObsType, dict]:
        super().reset(seed=seed)
        self.agent_positions_local = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        self.num_steps = 0
        self.update_positions_global()
        return self.get_observations(), {}

if MAIN:
    gym.envs.registration.register(id="MappoTest-v0", entry_point=MappoTest)
    env = gym.make("MappoTest-v0")
    assert env.observation_space.shape == (2, 4)
    assert env.action_space.shape == (2,)