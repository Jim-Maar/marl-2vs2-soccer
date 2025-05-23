# %% 
import itertools
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Callable
import pickle
import einops
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import wandb
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
from matplotlib.animation import FuncAnimation
from numpy.random import Generator
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import argparse

warnings.filterwarnings("ignore")

# %%

from utils import set_global_seeds
from environments.probes import Probe1, Probe2, Probe3, Probe4, Probe5, get_episode_data_from_infos
from environments.mappo_test import MappoTest
from environments.mappo_selfplay_test import MappoSelfplayTest
from environments.soccer import Soccer
from utils import prepare_atari_env
from utils import make_env
from plotly_utils import plot_cartpole_obs_and_dones

# Register our probes
for idx, probe in enumerate([Probe1, Probe2, Probe3, Probe4, Probe5]):
    gym.envs.registration.register(id=f"Probe{idx+1}-v0", entry_point=probe)

Arr = np.ndarray

directory = Path(__file__).parent

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

is_debugging = sys.gettrace() is not None
MAIN = __name__ == "__main__" or is_debugging

MODES = ["classic-control", "atari", "mujoco", "mappo-test", "soccer"]

RUN_NAME = "Real_Run"

STAMDART_REWARD_SPECIFICATION = {
    "goal": 200.0,
    "smoothness": 0.05,
    "stay_in_field": 0.05,
    "stay_own_half": 0.05,
    "base_negative": -0.15,
}

DIST_REWARD_SPECIFICATION = {
    "goal": 200.0,
    "player_distance": -0.05,
    "smoothness": 0.05,
    "stay_in_field": 0.05,
    "stay_own_half": 0.05,
    "base_negative": -0.15,
}

DIST_AND_PASSING_REWARD_SPECIFICATION = {
    "goal": 200.0,
    "distance_based_passing": 1.0,
    "player_distance": -0.05,
    "smoothness": 0.05,
    "stay_in_field": 0.05,
    "stay_own_half": 0.05,
    "base_negative": -0.15,
}

DIST_AND_PASSING_REWARD_SPECIFICATION_2 = {
    "goal": 100.0,
    "distance_based_passing": 4.0,
    "player_distance": -0.05,
    "smoothness": 0.05,
    "stay_in_field": 0.1,
    "stay_own_half": 0.05,
    "base_negative": -0.25,
}

DIST_AND_PASSING_REWARD_SPECIFICATION_3 = {
    "goal": 100.0,
    "distance_based_passing": 2.0,
    "player_distance": -0.05,
    "smoothness": 0.05,
    "stay_in_field": 0.1,
    "stay_own_half": 0.05,
    "base_negative": -0.25,
}

DIST_AND_PASSING_REWARD_SPECIFICATION_4 = {
    "goal": 100.0,
    "distance_based_passing": 1.5,
    "player_distance": -0.05,
    "smoothness": 0.05,
    "stay_in_field": 0.1,
    "stay_own_half": 0.05,
    "base_negative": -0.25,
}

DIST_AND_PASSING_AND_SHOOTING_REWARD_SPECIFICATION = {
    "goal": 200.0,
    "distance_based_passing": 2.0,
    "shooting": 0.075,
    "player_distance": -0.05,
    "smoothness": 0.05,
    "stay_in_field": 0.05,
    "stay_own_half": 0.05,
    "base_negative": -0.15,
}

REWARD_SPECIFICATIONS = {
    "standard": STAMDART_REWARD_SPECIFICATION,
    "dist": DIST_REWARD_SPECIFICATION,
    "dist_and_passing": DIST_AND_PASSING_REWARD_SPECIFICATION,
    "dist_and_passing_2": DIST_AND_PASSING_REWARD_SPECIFICATION_2,
    "dist_and_passing_3": DIST_AND_PASSING_REWARD_SPECIFICATION_3,
    "dist_and_passing_4": DIST_AND_PASSING_REWARD_SPECIFICATION_4,
    "dist_and_passing_and_shooting": DIST_AND_PASSING_AND_SHOOTING_REWARD_SPECIFICATION,
}

# %%
@dataclass
class PPOArgs:
    # Basic / global
    seed: int = 1
    env_id: str = "CartPole-v1"
    mode: Literal["classic-control", "atari", "mujoco", "mappo-test", "soccer"] = "classic-control"

    # Wandb / logging
    use_wandb: bool = False
    video_log_freq: int | None = None
    wandb_project_name: str = f"MAPPO_{RUN_NAME}"
    wandb_entity: str = None

    # Duration of different phases
    total_timesteps: int = 500_000
    num_envs: int = 4
    num_steps_per_rollout: int = 128
    num_minibatches: int = 4
    batches_per_learning_phase: int = 4

    # Optimization hyperparameters
    lr: float = 2.5e-4
    max_grad_norm: float = 0.5

    # RL hyperparameters
    gamma: float = 0.99

    # PPO-specific hyperparameters
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.25

    # multi-agent specific
    num_agents: int = 1
    team_size: int = 1
    num_teams_per_game: int = 1

    # population based training specific
    num_steps_per_checkpoint: int = 3000 # 100_000
    num_of_self_play_envs: int = 2
    expected_num_steps_per_team: int = 400 # 1200 * 4
    num_envs_per_team: int = 1

    # soccer speficic
    reward_specification: dict | None = None

    def __post_init__(self):
        self.batch_size = self.num_steps_per_rollout * (self.num_envs + self.num_of_self_play_envs) * self.team_size

        assert self.batch_size % self.num_minibatches == 0, "batch_size must be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.total_phases = self.total_timesteps // self.batch_size
        self.total_training_steps = self.total_phases * self.batches_per_learning_phase * self.num_minibatches

        self.video_save_path = directory / "videos"
        self.checkpoint_save_path = directory / "checkpoints"
        self.wandb_project_name = f"MAPPO_{RUN_NAME}"


ARG_HELP_STRINGS = dict(
    seed="seed of the experiment",
    env_id="the id of the environment",
    mode="can be 'classic-control', 'atari', 'mujoco', 'mappo-test' or 'soccer'",
    #
    use_wandb="if toggled, this experiment will be tracked with Weights and Biases",
    video_log_freq="if not None, we log videos this many episodes apart (so shorter episodes mean more frequent logging)",
    wandb_project_name="the name of this experiment (also used as the wandb project name)",
    wandb_entity="the entity (team) of wandb's project",
    #
    total_timesteps="total timesteps of the experiments",
    num_envs="number of synchronized vector environments in our `envs` object (this is N in the '37 Implementational Details' post)",
    num_steps_per_rollout="number of steps taken in the rollout phase (this is M in the '37 Implementational Details' post)",
    num_minibatches="the number of minibatches you divide each batch up into",
    batches_per_learning_phase="how many times you train on the full batch of data generated in each rollout phase",
    #
    lr="the learning rate of the optimizer",
    max_grad_norm="value used in gradient clipping",
    #
    gamma="the discount factor gamma",
    gae_lambda="the discount factor used in our GAE estimation",
    clip_coef="the epsilon term used in the clipped surrogate objective function",
    ent_coef="coefficient of entropy bonus term",
    vf_coef="cofficient of value loss function",
    #
    batch_size="N * M in the '37 Implementational Details' post (calculated from other values in PPOArgs)",
    minibatch_size="the size of a single minibatch we perform a gradient step on (calculated from other values in PPOArgs)",
    total_phases="total number of phases during training (calculated from other values in PPOArgs)",
    total_training_steps="total number of minibatches we will perform an update step on during training (calculated from other values in PPOArgs)",
    #
    num_agents="number of agents in the multi-agent case",
    num_teams_per_game="number of teams in the multi-agent case",
    #
    num_steps_per_checkpoint="number of steps per checkpoint in population based training",
    probability_of_self_play="probability of self-play in population based training",
    expected_num_steps_per_team="expected number of steps per team in population based training used to calculate the probability of swaping out teams",
    num_envs_per_team="number of environments per team in population based training. The first team is always the main team. and the second team can be older checkpoints",
)

def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_actor_and_critic(
    envs: gym.vector.SyncVectorEnv,
    mode: Literal["classic-control", "atari", "mujoco", "mappo-test", "soccer", "mappo-selfplay-test"] = "classic-control",
) -> tuple[nn.Module, nn.Module]:
    """
    Returns (actor, critic), the networks used for PPO, in one of 3 different modes.
    """
    assert mode in ["classic-control", "atari", "mujoco", "mappo-test", "soccer", "mappo-selfplay-test"]

    obs_global_shape = envs.single_observation_space.shape
    if mode == "mappo-test" or mode == "soccer" or mode == "mappo-selfplay-test":
        num_agents = obs_global_shape[0]
        obs_shape = obs_global_shape[1:]
    else:
        num_agents = 1
        obs_shape = obs_global_shape
    num_obs = np.array(obs_shape).prod()
    if isinstance(envs.single_action_space, gym.spaces.Discrete):
        num_actions = envs.single_action_space.n
    elif isinstance(envs.single_action_space, gym.spaces.MultiDiscrete):
        num_actions = np.array(envs.single_action_space.nvec)[0]
    else:
        num_actions = np.array(envs.single_action_space.shape).prod()

    # TODO: implement get_actor_and_critic_soccer and mappo-test
    if mode == "classic-control":
        actor, critic = get_actor_and_critic_classic(num_obs, num_actions)
    if mode == "atari":
        actor, critic = get_actor_and_critic_atari(obs_shape, num_actions)  # you'll implement these later
    if mode == "mujoco":
        actor, critic = get_actor_and_critic_mujoco(num_obs, num_actions)  # you'll implement these later
    if mode == "mappo-test" or mode == "mappo-selfplay-test":
        actor, critic = get_actor_and_critic_classic(num_obs, num_actions)  # you'll implement these later
    if mode == "soccer":
        actor, critic = get_actor_and_critic_soccer(num_obs, num_actions, NUM_LAYERS, NUM_HIDDEN_UNITS, ACTIVATION_FUNCTION)  # you'll implement these later
        # actor, critic = get_actor_and_critic_classic(num_obs, num_actions)

    return actor.to(device), critic.to(device)

# TODO: Do I need this?
def get_actor_and_critic_classic(num_obs: int, num_actions: int, num_hidden_layers: int = 2):
    """
    Returns (actor, critic) in the "classic-control" case, according to diagram above.
    """
    critic = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        *sum([[layer_init(nn.Linear(64, 64)), nn.Tanh()] for _ in range(num_hidden_layers-1)], []),
        layer_init(nn.Linear(64, 1), std=1.0),
    )
    actor = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        *sum([[layer_init(nn.Linear(64, 64)), nn.Tanh()] for _ in range(num_hidden_layers-1)], []),
        layer_init(nn.Linear(64, num_actions), std=0.01),
    )
    return actor, critic

def get_actor_and_critic_soccer(num_obs: int, num_actions: int, num_hidden_layers: int = 3, num_hidden_units: int = 64, activation_function : str = "GELU"):
    """
    Returns (actor, critic) in the "classic-control" case, according to diagram above.
    """
    if activation_function == "GELU":
        activation = nn.GELU()
    elif activation_function == "Tanh":
        activation = nn.Tanh()
    else:
        raise ValueError(f"Activation function {activation_function} not supported")
    critic = nn.Sequential(
        layer_init(nn.Linear(num_obs, num_hidden_units)),
        activation,
        *sum([[layer_init(nn.Linear(num_hidden_units, num_hidden_units)), activation] for _ in range(num_hidden_layers-1)], []),
        layer_init(nn.Linear(num_hidden_units, 1), std=1.0),
    )
    actor = nn.Sequential(
        layer_init(nn.Linear(num_obs, num_hidden_units)),
        activation,
        *sum([[layer_init(nn.Linear(num_hidden_units, num_hidden_units)), activation] for _ in range(num_hidden_layers-1)], []),
        layer_init(nn.Linear(num_hidden_units, num_actions), std=0.01),
    )
    return actor, critic


@t.inference_mode()
def compute_advantages(
    next_value: Float[Tensor, "num_envs num_agents"],
    next_terminated: Bool[Tensor, "num_envs num_agents"],
    rewards: Float[Tensor, "buffer_size num_envs num_agents"],
    values: Float[Tensor, "buffer_size num_envs num_agents"],
    terminated: Bool[Tensor, "buffer_size num_envs num_agents"],
    gamma: float,
    gae_lambda: float,
) -> Float[Tensor, "buffer_size num_envs num_agents"]:
    """
    Compute advantages using Generalized Advantage Estimation.
    """
    T = values.shape[0]
    terminated = terminated.float()
    next_terminated = next_terminated.float()

    # Get tensors of V(s_{t+1}) and d_{t+1} for all t = 0, 1, ..., T-1
    next_values = t.concat([values[1:], next_value[None, :, :]])
    next_terminated = t.concat([terminated[1:], next_terminated[None, :, :]])

    # Compute deltas: \delta_t = r_t + (1 - d_{t+1}) \gamma V(s_{t+1}) - V(s_t)
    deltas = rewards + gamma * next_values * (1.0 - next_terminated) - values

    # Compute advantages using the recursive formula, starting with advantages[T-1] = deltas[T-1] and working backwards
    advantages = t.zeros_like(deltas)
    advantages[-1] = deltas[-1]
    for s in reversed(range(T - 1)):
        advantages[s] = deltas[s] + gamma * gae_lambda * (1.0 - terminated[s + 1]) * advantages[s + 1]

    return advantages


def get_minibatch_indices(rng: Generator, batch_size: int, minibatch_size: int) -> list[np.ndarray]:
    """
    Return a list of length `num_minibatches`, where each element is an array of `minibatch_size` and the union of all
    the arrays is the set of indices [0, 1, ..., batch_size - 1] where `batch_size = num_steps_per_rollout * num_envs`.
    """
    assert batch_size % minibatch_size == 0
    num_minibatches = batch_size // minibatch_size
    indices = rng.permutation(batch_size).reshape(num_minibatches, minibatch_size)
    return list(indices)


@dataclass
class ReplayMinibatch:
    """
    Samples from the replay memory, converted to PyTorch for use in neural network training.

    Data is equivalent to (s_t, a_t, logpi(a_t|s_t), A_t, A_t + V(s_t), d_{t+1})
    """

    obs: Float[Tensor, "minibatch_size *obs_shape"]
    obs_global: Float[Tensor, "minibatch_size *obs_global_shape"]
    actions: Int[Tensor, "minibatch_size *action_shape"]
    logprobs: Float[Tensor, "minibatch_size"]
    advantages: Float[Tensor, "minibatch_size"]
    returns: Float[Tensor, "minibatch_size"]
    terminated: Bool[Tensor, "minibatch_size"]
    agent_idx: Int[Tensor, "minibatch_size"]

class ReplayMemory:
    """
    Contains buffer; has a method to sample from it to return a ReplayMinibatch object.
    """

    rng: Generator
    obs: Float[Arr, "buffer_size num_main_team_envs team_size *obs_shape"]
    obs_global: Float[Arr, "buffer_size num_main_team_envs team_size *obs_global_shape"]
    actions: Int[Arr, "buffer_size num_main_team_envs team_size *action_shape"]
    logprobs: Float[Arr, "buffer_size num_main_team_envs team_size"]
    values: Float[Arr, "buffer_size num_main_team_envs team_size"]
    rewards: Float[Arr, "buffer_size num_main_team_envs team_size"]
    terminated: Bool[Arr, "buffer_size num_main_team_envs team_size"]
    agent_idx: Int[Arr, "buffer_size num_main_team_envs team_size"]

    def __init__(
        self,
        num_envs: int,
        num_self_play_envs: int,
        num_agents: int,
        team_size: int,
        obs_shape: tuple,
        obs_global_shape: tuple,
        action_shape: tuple,
        batch_size: int,
        minibatch_size: int,
        batches_per_learning_phase: int,
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.num_self_play_envs = num_self_play_envs
        self.num_main_team_envs = num_envs + num_self_play_envs
        self.num_agents = num_agents
        self.team_size = team_size
        self.obs_shape = obs_shape
        self.obs_global_shape = obs_global_shape
        self.action_shape = action_shape
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.batches_per_learning_phase = batches_per_learning_phase
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Resets all stored experiences, ready for new ones to be added to memory."""
        num_steps_per_rollout = self.batch_size // (self.num_main_team_envs * self.team_size)
        self.obs = np.empty((0, self.num_main_team_envs, self.team_size, *self.obs_shape), dtype=np.float32)
        self.obs_global = np.empty((0, self.num_main_team_envs, self.team_size, *self.obs_global_shape), dtype=np.float32)
        self.actions = np.empty((0, self.num_main_team_envs, self.team_size, *self.action_shape), dtype=np.int32)
        self.logprobs = np.empty((0, self.num_main_team_envs, self.team_size), dtype=np.float32)
        self.values = np.empty((0, self.num_main_team_envs, self.team_size), dtype=np.float32)
        self.rewards = np.empty((0, self.num_main_team_envs, self.team_size), dtype=np.float32)
        self.terminated = np.empty((0, self.num_main_team_envs, self.team_size), dtype=bool)
        self.agent_idx = np.concatenate((einops.repeat(np.arange(self.team_size), "team_size -> num_steps_per_rollout num_envs team_size", num_steps_per_rollout=num_steps_per_rollout, num_envs=self.num_envs), einops.repeat(np.arange(self.team_size, self.num_agents), "team_size -> num_steps_per_rollout num_self_play_envs team_size", num_steps_per_rollout=num_steps_per_rollout, num_self_play_envs=self.num_self_play_envs)), axis=1)

    def add(
        self,
        obs: Float[Arr, "num_envs num_agents *obs_shape"],
        obs_global: Float[Arr, "num_envs num_agents *obs_global_shape"],
        actions: Int[Arr, "num_envs num_agents *action_shape"],
        logprobs: Float[Arr, "num_envs num_agents"],
        values: Float[Arr, "num_envs num_agents"],
        rewards: Float[Arr, "num_envs num_agents"],
        terminated: Bool[Arr, "num_envs num_agents"],
    ) -> None:
        """Add a batch of transitions to the replay memory."""
        # Check shapes & datatypes
        for data, expected_shape in zip(
            [obs, obs_global, actions, logprobs, values, rewards, terminated], [self.obs_shape, self.obs_global_shape, self.action_shape, (), (), (), ()]
        ):
            assert isinstance(data, np.ndarray)
            assert data.shape == (self.num_main_team_envs, self.team_size, *expected_shape)

        # Add data to buffer (not slicing off old elements)
        self.obs = np.concatenate((self.obs, obs[None, :, :]))
        self.obs_global = np.concatenate((self.obs_global, obs_global[None, :, :]))
        self.actions = np.concatenate((self.actions, actions[None, :, :]))
        self.logprobs = np.concatenate((self.logprobs, logprobs[None, :, :]))
        self.values = np.concatenate((self.values, values[None, :, :]))
        self.rewards = np.concatenate((self.rewards, rewards[None, :, :]))
        self.terminated = np.concatenate((self.terminated, terminated[None, :, :]))

    def get_minibatches(
        self, next_value: Tensor, next_terminated: Tensor, gamma: float, gae_lambda: float
    ) -> list[ReplayMinibatch]:
        """
        Returns a list of minibatches. Each minibatch has size `minibatch_size`, and the union over all minibatches is
        `batches_per_learning_phase` copies of the entire replay memory.
        """
        # Convert everything to tensors on the correct device
        obs, obs_global, actions, logprobs, values, rewards, terminated, agent_idx = (
            t.tensor(x, device=device)
            for x in [self.obs, self.obs_global, self.actions, self.logprobs, self.values, self.rewards, self.terminated, self.agent_idx]
        )

        # Compute advantages & returns
        advantages = compute_advantages(next_value, next_terminated, rewards, values, terminated, gamma, gae_lambda)
        returns = advantages + values

        # Return a list of minibatches
        minibatches = []
        for _ in range(self.batches_per_learning_phase):
            for indices in get_minibatch_indices(self.rng, self.batch_size, self.minibatch_size):
                minibatches.append(
                    ReplayMinibatch(
                        *[
                            data.flatten(0, 2)[indices]
                            for data in [obs, obs_global, actions, logprobs, advantages, returns, terminated, agent_idx]
                        ]
                    )
                )

        # Reset memory (since we only need to call this method once per learning phase)
        self.reset()

        return minibatches

def get_team_rewards(reward: Float[Arr, "num_envs"], infos: dict, num_agents: int) -> Float[Arr, "num_envs num_teams_per_game"]:
    # TODO: test this for more than two agents ...
    if num_agents == 1:
        return reward[:, None]
    if num_agents == 2:
        if "final_info" in infos:
            assert "other_reward" in infos["final_info"][0]
            other_reward = np.array([info["other_reward"] for info in infos["final_info"]])[:, None]
        else:
            assert "other_reward" in infos
            other_reward = infos["other_reward"][:, None]
    if num_agents > 2:
        if "final_info" in infos:
            other_reward = []
            for env_idx in range(len(infos["final_info"])):
                if infos["final_info"][env_idx] is None:
                    other_reward.append(infos["other_reward"][env_idx])
                else:
                    other_reward.append(infos["final_info"][env_idx]["other_reward"])
            other_reward = np.stack(other_reward, axis=0)
        else:
            assert "other_reward" in infos
            other_reward = np.stack(infos["other_reward"], axis=0)
    return np.concatenate([reward[:, None], other_reward], axis=1)

# %%
class PPOPopulation:
    def __init__(
        self, 
        envs: gym.vector.SyncVectorEnv, 
        actor: nn.Module,
        critic: nn.Module,
        memory: ReplayMemory,
        args: PPOArgs,
        run_name: str,
        num_agents: int, 
        num_teams_per_game: int,
        team_size: int,
        seed: int,
    ):
        super().__init__()
        self.envs = envs
        self.memory = memory
        self.rng = np.random.default_rng(seed)

        self.num_agents = num_agents
        self.num_teams_per_game = num_teams_per_game
        self.team_size = team_size

        self.run_name = run_name
        self.args = args

        self.step = 0  # Tracking number of steps taken (across all environments)
        self.num_of_self_play_envs = self.args.num_of_self_play_envs
        self.num_envs_per_team = self.args.num_envs_per_team
        self.num_steps_per_checkpoint = self.args.num_steps_per_checkpoint
        self.expected_num_steps_per_team = self.args.expected_num_steps_per_team
        
        # Original line that causes the error
        self.next_obs = t.tensor(envs.reset()[0], device=device, dtype=t.float)  # need starting obs (in tensor form)
        if self.num_agents == 1:
            self.next_obs = self.next_obs.unsqueeze(dim=1)
        self.next_terminated = t.zeros((envs.num_envs, self.num_agents), device=device, dtype=t.bool)  # need starting termination=False

        self.main_team = PPOTeam(actor, critic, self.envs, self.args.mode, self.args.checkpoint_save_path, self.run_name)
        self.initialize_teams()

    def initialize_teams(self):
        main_team_copy = self.main_team.copy_and_save()
        self.other_teams = [main_team_copy]
        assert (self.envs.num_envs - self.num_of_self_play_envs) % self.num_envs_per_team == 0
        self.num_active_other_teams = (self.envs.num_envs - self.num_of_self_play_envs) // self.num_envs_per_team
        self.active_other_teams = [main_team_copy for _ in range(self.num_active_other_teams)]

    def play_step(self):
        # get actions, logprobs and values from agent
        obs = self.next_obs
        terminated = self.next_terminated
        main_team_obs = t.concat((obs[:, :self.team_size], obs[self.num_of_self_play_envs:, self.team_size:]), dim=0)
        main_team_actions, main_team_logprobs = self.main_team.get_actions(main_team_obs)
        main_team_values = self.main_team.get_values(main_team_obs)
        other_team_actions_list = []
        for i in range(self.num_active_other_teams):
            first_env_idx = i * self.num_envs_per_team # + self.num_of_self_play_envs
            last_env_idx = first_env_idx + self.num_envs_per_team
            other_team_obs = obs[first_env_idx:last_env_idx, :self.team_size]
            other_team_actions, _ = self.active_other_teams[i].get_actions(other_team_obs)
            other_team_actions_list.append(other_team_actions)
        other_team_actions = t.concat(other_team_actions_list, dim=0) if len(other_team_actions_list) > 0 else t.zeros((0, 2)).to(device)
        first_team_actions = main_team_actions[:self.envs.num_envs]
        second_team_actions = t.concat((other_team_actions, main_team_actions[self.envs.num_envs:]), dim=0)
        actions = t.concat((first_team_actions, second_team_actions), dim=1)
        next_obs, reward, next_terminated, next_truncated, infos = self.envs.step(actions)
        reward = get_team_rewards(reward, infos, self.num_agents)
        main_team_rewards = np.concatenate((reward[:, :self.team_size], reward[self.num_of_self_play_envs:, self.team_size:]), axis=0)
        next_terminated = einops.repeat(np.concatenate((next_terminated, next_terminated[self.num_of_self_play_envs:]), axis=0), "num_main_team_envs -> num_main_team_envs num_agents_in_main_team", num_agents_in_main_team=self.team_size)
        next_truncated = einops.repeat(np.concatenate((next_truncated, next_truncated[self.num_of_self_play_envs:]), axis=0), "num_main_team_envs -> num_main_team_envs num_agents_in_main_team", num_agents_in_main_team=self.team_size)
        # add to memory
        self.memory.add(
            obs = main_team_obs.cpu().numpy(),
            obs_global = main_team_obs.cpu().numpy(),
            actions = main_team_actions.cpu().numpy(),
            logprobs = main_team_logprobs.cpu().numpy(),
            values = main_team_values.cpu().numpy(),
            rewards = main_team_rewards,
            terminated = next_terminated,
        )
        # update next_obs and next_terminated
        self.next_obs = t.from_numpy(next_obs).to(device, dtype=t.float)
        if self.num_agents == 1:
            self.next_obs = self.next_obs.unsqueeze(dim=1)
        self.next_terminated = t.from_numpy(next_terminated).to(device, dtype=t.float)
        last_step = self.step
        self.step += (self.envs.num_envs + self.num_of_self_play_envs) * self.team_size
        # swap teams (next ...)
        self.swap_teams()
        # every num_steps_per_checkpoint steps, add a checkpoint
        if self.step // self.num_steps_per_checkpoint > (last_step // self.num_steps_per_checkpoint):
            self.save_checkpoint()
        return infos

    def save_checkpoint(self):
        checkpoint_team = self.main_team.copy_and_save()
        self.other_teams.append(checkpoint_team)

    def swap_teams(self):
        random_nums = self.rng.random(size=(self.num_active_other_teams,))
        to_swap = random_nums <= 1 / self.expected_num_steps_per_team # self.num_envs_per_team * self.num_agents / self.expected_num_steps_per_team
        indices_to_swap = np.arange(self.num_active_other_teams)[to_swap]
        for i in indices_to_swap:
            random_team = self.rng.choice(self.other_teams)
            self.active_other_teams[i] = random_team

    def get_minibatches(self, gamma: float, gae_lambda: float) -> list[ReplayMinibatch]:
        """
        Gets minibatches from the replay memory, and resets the memory
        """
        with t.inference_mode():
            main_team_obs = t.concat((self.next_obs[:, :self.team_size], self.next_obs[self.num_of_self_play_envs:, self.team_size:]), dim=0)
            main_team_values = self.main_team.get_values(main_team_obs)
            next_value = main_team_values
        minibatches = self.memory.get_minibatches(next_value, self.next_terminated, gamma, gae_lambda)
        self.memory.reset()
        return minibatches

class PPOTeam:
    def __init__(self, actor: nn.Module, critic: nn.Module, envs: gym.vector.SyncVectorEnv, mode: str, checkpoint_save_path: str, run_name: str):
        self.actor = actor
        self.critic = critic
        self.envs = envs
        self.mode = mode
        self.checkpoint_save_path = checkpoint_save_path
        self.run_name = run_name

    def copy_and_save(self):
        """Create a deep copy of this team."""
        new_actor, new_critic = get_actor_and_critic(self.envs, self.mode)
        new_actor.load_state_dict(self.actor.state_dict())
        new_critic.load_state_dict(self.critic.state_dict())

        path = self.checkpoint_save_path / self.run_name
        path.mkdir(parents=True, exist_ok=True)
        num_files = len(list(path.glob("*.pth")))
        id = num_files // 2

        t.save(new_actor, f"{path}/actor_{id}.pth")
        t.save(new_critic, f"{path}/critic_{id}.pth")
        
        return PPOTeam(new_actor, new_critic, self.envs, self.mode, self.checkpoint_save_path, self.run_name)
        
    def get_actions(self, obs: Float[Arr, "num_envs num_agents *obs_shape"]) -> tuple[Float[Arr, "num_envs num_agents *action_shape"], Float[Arr, "num_envs num_agents *action_shape"]]:
        with t.inference_mode():
            logits = self.actor(obs)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)
        return actions, logprobs

    def get_values(self, obs: Float[Arr, "num_envs num_agents *obs_shape"]) -> Float[Arr, "num_envs num_agents"]:
        with t.inference_mode():
            values = self.critic(obs).squeeze(dim=-1)
        return values

def calc_clipped_surrogate_objective(
    probs: Categorical,
    mb_action: Int[Tensor, "minibatch_size"],
    mb_advantages: Float[Tensor, "minibatch_size"],
    mb_logprobs: Float[Tensor, "minibatch_size"],
    clip_coef: float,
    eps: float = 1e-8,
) -> Float[Tensor, ""]:
    """Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    probs:
        a distribution containing the actor's unnormalized logits of shape (minibatch_size, num_actions)
    mb_action:
        what actions actions were taken in the sampled minibatch
    mb_advantages:
        advantages calculated from the sampled minibatch
    mb_logprobs:
        logprobs of the actions taken in the sampled minibatch (according to the old policy)
    clip_coef:
        amount of clipping, denoted by epsilon in Eq 7.
    eps:
        used to add to std dev of mb_advantages when normalizing (to avoid dividing by zero)
    """
    assert mb_action.shape == mb_advantages.shape == mb_logprobs.shape
    logits_diff = probs.log_prob(mb_action) - mb_logprobs

    prob_ratio = t.exp(logits_diff)

    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps)

    non_clipped = prob_ratio * mb_advantages
    clipped = t.clip(prob_ratio, 1 - clip_coef, 1 + clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()


def calc_value_function_loss(
    values: Float[Tensor, "minibatch_size"], mb_returns: Float[Tensor, "minibatch_size"], vf_coef: float
) -> Float[Tensor, ""]:
    """Compute the value function portion of the loss function.

    values:
        the value function predictions for the sampled minibatch (using the updated critic network)
    mb_returns:
        the target for our updated critic network (computed as `advantages + values` from the old network)
    vf_coef:
        the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    """
    assert values.shape == mb_returns.shape

    return vf_coef * (values - mb_returns).pow(2).mean()


def calc_entropy_bonus(dist: Categorical, ent_coef: float):
    """Return the entropy bonus term, suitable for gradient ascent.

    dist:
        the probability distribution for the current policy
    ent_coef:
        the coefficient for the entropy loss, which weights its contribution to the overall objective function. Denoted by c_2 in the paper.
    """
    return ent_coef * dist.entropy().mean()

class PPOScheduler:
    def __init__(self, optimizer: Optimizer, initial_lr: float, end_lr: float, total_phases: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.total_phases = total_phases
        self.n_step_calls = 0

    def step(self):
        """Implement linear learning rate decay so that after `total_phases` calls to step, the learning rate is end_lr.

        Do this by directly editing the learning rates inside each param group (i.e. `param_group["lr"] = ...`), for each param
        group in `self.optimizer.param_groups`.
        """
        self.n_step_calls += 1
        frac = self.n_step_calls / self.total_phases
        assert frac <= 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)


def make_optimizer(
    actor: nn.Module, critic: nn.Module, total_phases: int, initial_lr: float, end_lr: float = 0.0
) -> tuple[optim.Adam, PPOScheduler]:
    """
    Return an appropriately configured Adam with its attached scheduler.
    """
    optimizer = optim.AdamW(
        itertools.chain(actor.parameters(), critic.parameters()), lr=initial_lr, eps=1e-5, maximize=True
    )
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, total_phases)
    return optimizer, scheduler

def make_env_wrapper(idx: int, run_name: str, args: PPOArgs, mode: str) -> gym.Env:
    if mode == "mappo-test":
        return MappoTest
    else:
        return make_env(idx=idx, run_name=run_name, **args.__dict__)

class PPOTrainer:
    def __init__(self, args: PPOArgs):
        set_global_seeds(args.seed)
        self.args = args
        self.run_name = f"{args.env_id}__{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        # self.envs = gym.vector.SyncVectorEnv(
        #     [make_env(idx=idx, run_name=self.run_name, **args.__dict__) for idx in range(args.num_envs)]
        # )
        self.envs = gym.vector.SyncVectorEnv([make_env_wrapper(idx=idx, run_name=self.run_name, args=args, mode=args.mode) for idx in range(args.num_envs)])

        self.num_agents = self.args.num_agents
        self.num_teams_per_game = self.args.num_teams_per_game
        # self.team_size = self.num_agents // self.num_teams_per_game
        self.team_size = self.args.team_size

        # Define some basic variables from our environment
        self.num_envs = self.envs.num_envs
        self.action_shape = self.envs.single_action_space.shape
        if self.args.num_agents > 1:
            self.action_shape = self.action_shape[1:]
        self.obs_global_shape = self.envs.single_observation_space.shape
        if self.args.num_agents > 1:
            self.obs_global_shape = self.obs_global_shape[1:]

        # Create our replay memory
        self.memory = ReplayMemory(
            self.num_envs,
            self.args.num_of_self_play_envs,
            self.num_agents,
            self.team_size,
            self.obs_global_shape,
            self.obs_global_shape,
            self.action_shape,
            args.batch_size,
            args.minibatch_size,
            args.batches_per_learning_phase,
            args.seed,
        )

        # Create our networks & optimizer
        self.actor, self.critic = get_actor_and_critic(self.envs, mode=args.mode)
        self.optimizer, self.scheduler = make_optimizer(self.actor, self.critic, args.total_training_steps, args.lr)

        # get the get_obs_for_agent function
        # get_obs_for_agent = get_obs_for_agent_function(mode=args.mode)

        # Create our population
        self.population = PPOPopulation(
            self.envs,
            self.actor,
            self.critic,
            self.memory,
            self.args,
            self.run_name,
            self.num_agents,
            self.num_teams_per_game,
            self.team_size,
            self.args.seed,
        )

    def rollout_phase(self) -> dict | None:
        """
        This function populates the memory with a new set of experiences, using `self.population.play_step` to step through
        the environment. It also returns a dict of data which you can include in your progress bar postfix.
        """
        data = None
        t0 = time.time()

        for step in range(self.args.num_steps_per_rollout):
            # Play a step, returning the infos dict (containing information for each environment)
            infos = self.population.play_step()

            # Get data from environments, and log it if some environment did actually terminate
            new_data = get_episode_data_from_infos(infos)
            if new_data is not None:
                data = new_data
                if self.args.use_wandb:
                    wandb.log(new_data, step=self.population.step)

        if self.args.use_wandb:
            wandb.log(
                {"SPS": (self.args.num_steps_per_rollout * self.num_envs) / (time.time() - t0)}, step=self.population.step
            )

        return data

    def learning_phase(self) -> None:
        """
        This function does the following:
            - Generates minibatches from memory
            - Calculates the objective function, and takes an optimization step based on it
            - Clips the gradients (see detail #11)
            - Steps the learning rate scheduler
        """
        minibatches = self.population.get_minibatches(self.args.gamma, self.args.gae_lambda)
        for minibatch in minibatches:
            objective_fn = self.compute_ppo_objective(minibatch)
            objective_fn.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), self.args.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.scheduler.step()

    def compute_ppo_objective(self, minibatch: ReplayMinibatch) -> Float[Tensor, ""]:
        """
        Handles learning phase for a single minibatch. Returns objective function to be maximized.
        """
        agent_idx = minibatch.agent_idx
        team_idx = agent_idx // self.team_size
        first_agent_idx = team_idx * self.team_size
        indeces = t.arange(len(minibatch.obs_global))
        # team_obs = minibatch.obs_global[indeces, first_agent_idx]
    
        logits = self.actor(minibatch.obs)
        dist = Categorical(logits=logits)
        values = self.critic(minibatch.obs_global).squeeze()

        clipped_surrogate_objective = calc_clipped_surrogate_objective(
            dist, minibatch.actions, minibatch.advantages, minibatch.logprobs, self.args.clip_coef
        )
        value_loss = calc_value_function_loss(values, minibatch.returns, self.args.vf_coef)
        entropy_bonus = calc_entropy_bonus(dist, self.args.ent_coef)

        total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus

        with t.inference_mode():
            newlogprob = dist.log_prob(minibatch.actions)
            logratio = newlogprob - minibatch.logprobs
            ratio = logratio.exp()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs = [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
        if self.args.use_wandb:
            wandb.log(
                dict(
                    total_steps=self.population.step,
                    values=values.mean().item(),
                    lr=self.scheduler.optimizer.param_groups[0]["lr"],
                    value_loss=value_loss.item(),
                    clipped_surrogate_objective=clipped_surrogate_objective.item(),
                    entropy=entropy_bonus.item(),
                    approx_kl=approx_kl,
                    clipfrac=np.mean(clipfracs),
                ),
                step=self.population.step,
            )

        return total_objective_function

    def train(self) -> None:
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
                monitor_gym=self.args.video_log_freq is not None,
            )
            wandb.watch([self.actor, self.critic], log="all", log_freq=50)

        pbar = tqdm(range(self.args.total_phases))
        last_logged_time = time.time()  # so we don't update the progress bar too much

        for phase in pbar:
            data = self.rollout_phase()
            if data is not None and time.time() - last_logged_time > 0.5:
                last_logged_time = time.time()
                pbar.set_postfix(phase=phase, **data)

            # if phase >= 40:
            #     print("phase", phase)
            self.learning_phase()

        self.envs.close()
        if self.args.use_wandb:
            wandb.finish()

gym.envs.registration.register(id="Soccer-v0", entry_point=Soccer, apply_api_compatibility=False)
if MAIN:
    # Parse command line arguments for neural network architecture
    parser = argparse.ArgumentParser(description='Train PPO agents for soccer')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of hidden layers in the neural network')
    parser.add_argument('--num_hidden_units', type=int, default=128, help='Number of hidden units per layer')
    parser.add_argument('--activation', type=str, default="GELU", choices=["GELU", "Tanh"], help='Activation function to use')
    parser.add_argument('--reward_specification', type=str, default="dist_and_passing_and_shooting", choices=list(REWARD_SPECIFICATIONS.keys()), help='Reward specification to use')
    
    # Parse args and set global constants
    nn_args = parser.parse_args()
    NUM_LAYERS = nn_args.num_layers
    NUM_HIDDEN_UNITS = nn_args.num_hidden_units
    ACTIVATION_FUNCTION = nn_args.activation
    REWARD_SPECIFICATION = REWARD_SPECIFICATIONS[nn_args.reward_specification]

    BASE_RUN_NAME = ""
    RUN_NAME = f"{BASE_RUN_NAME}_{nn_args.reward_specification}_{NUM_LAYERS}_layer_{NUM_HIDDEN_UNITS}_hidden_units_{ACTIVATION_FUNCTION}"
    
    print(f"RUN_NAME: {RUN_NAME}")
    print(f"REWARD_SPECIFICATION: {REWARD_SPECIFICATION}")
    
    args = PPOArgs(
        env_id="Soccer-v0",
        mode="soccer",
        use_wandb=False,
        video_log_freq=50,
        num_envs=4,
        num_agents=4,
        team_size = 2,
        num_teams_per_game=2,
        total_timesteps=500_000_000,
        num_steps_per_rollout = 4096,
        num_minibatches = 16,
        batches_per_learning_phase = 8,
        num_steps_per_checkpoint = 100_000,
        num_of_self_play_envs = 2,
        expected_num_steps_per_team = 900,
        num_envs_per_team = 1,
        reward_specification = REWARD_SPECIFICATION,
    )
    trainer = PPOTrainer(args)
    trainer.train()
    # save trainer with pickle
    with open("soccer_trainer.pkl", "wb") as f:
        pickle.dump(trainer, f)

'''
def test_probe(probe_idx: int):
    """
    Tests a probe environment by training a network on it & verifying that the value functions are
    in the expected range.
    """
    # Train our network
    args = PPOArgs(
        env_id=f"Probe{probe_idx}-v0",
        wandb_project_name=f"test-probe-{probe_idx}",
        total_timesteps=[7500, 7500, 12500, 20000, 20000][probe_idx - 1],
        lr=0.001,
        video_log_freq=None,
        use_wandb=False,
        num_envs=4,
    )
    trainer = PPOTrainer(args)
    trainer.train()
    population = trainer.population

    # Get the correct set of observations, and corresponding values we expect
    obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
    expected_value_for_probes = [[[1.0]], [[-1.0], [+1.0]], [[args.gamma], [1.0]], [[1.0]], [[1.0], [1.0]]]
    expected_probs_for_probes = [None, None, None, [[0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
    tolerances = [1e-3, 1e-3, 1e-3, 2e-3, 2e-3]
    obs = t.tensor(obs_for_probes[probe_idx - 1]).to(device)

    # Calculate the actual value & probs, and verify them
    with t.inference_mode():
        value = population.critic(obs)
        probs = population.actor(obs).softmax(-1)
    expected_value = t.tensor(expected_value_for_probes[probe_idx - 1]).to(device)
    t.testing.assert_close(value, expected_value, atol=tolerances[probe_idx - 1], rtol=0)
    expected_probs = expected_probs_for_probes[probe_idx - 1]
    if expected_probs is not None:
        t.testing.assert_close(probs, t.tensor(expected_probs).to(device), atol=tolerances[probe_idx - 1], rtol=0)
    print("Probe tests passed!\n")


# %%
gym.envs.registration.register(id="MappoTest-v0", entry_point=MappoTest, apply_api_compatibility=False)
def test_mappo():
    args = PPOArgs(
        env_id="MappoTest-v0",
        mode="mappo-test",
        num_agents=2,
        team_size = 2,
        total_timesteps=70_000,
        use_wandb=False,
        gamma=0.0,
        num_envs=4,
    )
    trainer = PPOTrainer(args)
    trainer.train()
    population = trainer.population
    obs = t.tensor([[1.0, 0.0, 1.0, 0.0]]).to(device)
    with t.inference_mode():
        value = population.critic(obs)
    print(value)
    expected_value = t.tensor([[1.0]]).to(device)
    t.testing.assert_close(value, expected_value, atol=1e-2, rtol=0)
    print("Mappo test passed!")


gym.envs.registration.register(id="MappoSelfplayTest-v0", entry_point=MappoSelfplayTest, apply_api_compatibility=False)
def test_mappo_selfplay():
    args = PPOArgs(
        env_id="MappoSelfplayTest-v0",
        mode="mappo-selfplay-test",
        num_agents=4,
        team_size = 2,
        num_teams_per_game=2,
        total_timesteps=200_000,
        use_wandb=False,
        gamma=0.0,
        num_envs=4,
    )
    trainer = PPOTrainer(args)
    trainer.train()
    population = trainer.population
    obs = t.tensor([[0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]]).to(device)
    with t.inference_mode():
        value = population.main_team.critic(obs)
    print(value)
    expected_value = t.tensor([[1.0]]).to(device)
    t.testing.assert_close(value, expected_value, atol=5 * 1e-2, rtol=0)

# if MAIN:
#     for probe_idx in range(1, 6):
#         test_probe(probe_idx)

# if MAIN:
#     test_mappo()

# if MAIN:
#     test_mappo_selfplay()
'''