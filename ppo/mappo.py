# %% 
import itertools
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Callable

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

warnings.filterwarnings("ignore")

# %%

from utils import set_global_seeds
from environments.probes import Probe1, Probe2, Probe3, Probe4, Probe5, get_episode_data_from_infos
from environments.mappo_test import MappoTest
from environments.mappo_selfplay_test import MappoSelfplayTest
from utils import prepare_atari_env
from utils import arg_help, make_env
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
    wandb_project_name: str = "PPOCartPole"
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
    num_teams: int = 1

    def __post_init__(self):
        self.batch_size = self.num_steps_per_rollout * self.num_envs

        assert self.batch_size % self.num_minibatches == 0, "batch_size must be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.total_phases = self.total_timesteps // self.batch_size
        self.total_training_steps = self.total_phases * self.batches_per_learning_phase * self.num_minibatches

        self.video_save_path = directory / "videos"


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
    num_teams="number of teams in the multi-agent case",
)


if MAIN:
    args = PPOArgs(num_minibatches=2)  # changing this also changes minibatch_size and total_training_steps
    arg_help(args)


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
        actor, critic = get_actor_and_critic_mappo_test(num_obs, num_actions)  # you'll implement these later
    if mode == "soccer":
        actor, critic = get_actor_and_critic_soccer(num_obs, num_actions, num_agents)  # you'll implement these later

    return actor.to(device), critic.to(device)


def get_actor_and_critic_mappo_test(num_obs: int, num_actions: int):
    actor = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, num_actions), std=0.01),
    )
    critic = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 1), std=1.0),
    )
    return actor, critic

"""def get_ovs_for_agent_standart(global_obs: Arr, agent_idx: int) -> Arr:
    return global_obs

def get_obs_for_agent_function(mode):
    if mode == "classic-control" or mode == "atari" or mode == "mujoco":
        return get_ovs_for_agent_standart
    if mode == "mappo-test":
        return get_ovs_for_agent_mappo_test
    if mode == "soccer":
        return get_ovs_for_agent_soccer"""

# TODO: Do I need this?
def get_actor_and_critic_classic(num_obs: int, num_actions: int):
    """
    Returns (actor, critic) in the "classic-control" case, according to diagram above.
    """
    critic = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 1), std=1.0),
    )

    actor = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, num_actions), std=0.01),
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
    obs: Float[Arr, "buffer_size num_envs num_agents *obs_shape"]
    obs_global: Float[Arr, "buffer_size num_envs num_agents *obs_global_shape"]
    actions: Int[Arr, "buffer_size num_envs num_agents *action_shape"]
    logprobs: Float[Arr, "buffer_size num_envs num_agents"]
    values: Float[Arr, "buffer_size num_envs num_agents"]
    rewards: Float[Arr, "buffer_size num_envs num_agents"]
    terminated: Bool[Arr, "buffer_size num_envs num_agents"]
    agent_idx: Int[Arr, "buffer_size num_envs num_agents"]

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        obs_shape: tuple,
        obs_global_shape: tuple,
        action_shape: tuple,
        batch_size: int,
        minibatch_size: int,
        batches_per_learning_phase: int,
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.num_agents = num_agents
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
        self.obs = np.empty((0, self.num_envs, self.num_agents, *self.obs_shape), dtype=np.float32)
        self.obs_global = np.empty((0, self.num_envs, self.num_agents, *self.obs_global_shape), dtype=np.float32)
        self.actions = np.empty((0, self.num_envs, self.num_agents, *self.action_shape), dtype=np.int32)
        self.logprobs = np.empty((0, self.num_envs, self.num_agents), dtype=np.float32)
        self.values = np.empty((0, self.num_envs, self.num_agents), dtype=np.float32)
        self.rewards = np.empty((0, self.num_envs, self.num_agents), dtype=np.float32)
        self.terminated = np.empty((0, self.num_envs, self.num_agents), dtype=bool)
        self.agent_idx = einops.repeat(np.arange(self.num_agents), "num_agents -> minibatch_size num_envs num_agents", minibatch_size=self.minibatch_size, num_envs=self.num_envs)

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
            [obs, obs_global, actions, logprobs, values, rewards, terminated], [(self.num_agents, *self.obs_shape), (self.num_agents, *self.obs_global_shape), (self.num_agents, *self.action_shape), (self.num_agents, ), (self.num_agents, ), (self.num_agents, ), (self.num_agents, )]
        ):
            assert isinstance(data, np.ndarray)
            assert data.shape == (self.num_envs, *expected_shape)

        # Add data to buffer (not slicing off old elements)
        self.obs = np.concatenate((self.obs, obs[None, :, :]))
        self.obs_global = np.concatenate((self.obs_global, obs_global[None, :, :]))
        self.actions = np.concatenate((self.actions, actions[None, :, :]))
        self.logprobs = np.concatenate((self.logprobs, logprobs[None, :, :]))
        self.values = np.concatenate((self.values, values[None, :, :]))
        self.rewards = np.concatenate((self.rewards, rewards[None, :, :]))
        self.terminated = np.concatenate((self.terminated, terminated[None, :, :]))
        # Create array filled with agent_idx and concatenate
        # agent_idx_array = np.full((self.num_envs), fill_value=agent_idx, dtype=np.int32)
        # self.agent_idx = np.concatenate((self.agent_idx, agent_idx_array[None, :]))

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
    
"""
if MAIN:
    num_steps_per_rollout = 128
    num_envs = 2
    batch_size = num_steps_per_rollout * num_envs  # 256

    minibatch_size = 128
    num_minibatches = batch_size // minibatch_size  # 2

    batches_per_learning_phase = 2

    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", i, i, "test") for i in range(num_envs)])
    memory = ReplayMemory(num_envs, (4,), (), batch_size, minibatch_size, batches_per_learning_phase)

    logprobs = values = np.zeros(envs.num_envs)  # dummy values, just so we can see demo of plot
    obs, _ = envs.reset()

    for i in range(args.num_steps_per_rollout):
        # Choose random action, and take a step in the environment
        actions = envs.action_space.sample()
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # Add experience to memory
        memory.add(obs, actions, logprobs, values, rewards, terminated)
        obs = next_obs

    plot_cartpole_obs_and_dones(
        memory.obs,
        memory.terminated,
        title="Current obs s<sub>t</sub><br>Dotted lines indicate d<sub>t+1</sub> = 1, solid lines are environment separators",
    )

    next_value = next_done = t.zeros(envs.num_envs).to(device)  # dummy values, just so we can see demo of plot
    minibatches = memory.get_minibatches(next_value, next_done, gamma=0.99, gae_lambda=0.95)

    plot_cartpole_obs_and_dones(
        minibatches[0].obs.cpu(),
        minibatches[0].terminated.cpu(),
        title="Current obs (sampled)<br>this is what gets fed into our model for training",
    )
"""

def get_team_rewards(reward: Float[Arr, "num_envs"], infos: dict, num_agents: int) -> Float[Arr, "num_envs num_teams"]:
    # TODO: test this for more than two agents ...
    if num_agents == 1:
        return reward[:, None]
    if num_agents == 2:
        if "final_info" in infos:
            assert "other_reward" in infos["final_info"][0]
            other_reward = np.array([info["other_reward"] for info in infos["final_info"]])
        else:
            assert "other_reward" in infos
            other_reward = infos["other_reward"][:, None]
    if num_agents > 2:
        if "final_info" in infos:
            assert "other_reward" in infos["final_info"][0]
            other_reward = np.array([info["other_reward"] for info in infos["final_info"]])
        else:
            assert "other_reward" in infos
            other_reward = np.stack(infos["other_reward"], axis=0)
    return np.concatenate([reward[:, None], other_reward], axis=1)

# %%
class PPOAgents:
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv, actor: nn.Module, critic: nn.Module, memory: ReplayMemory, num_agents: int, num_teams: int):
        super().__init__()
        self.envs = envs
        self.actor = actor
        self.critic = critic
        self.memory = memory
        self.num_agents = num_agents
        self.num_teams = num_teams
        self.team_size = num_agents // num_teams
        # self.get_obs_for_agent = get_obs_for_agent

        self.step = 0  # Tracking number of steps taken (across all environments)
        self.next_obs = t.tensor(envs.reset()[0], device=device, dtype=t.float)  # need starting obs (in tensor form)
        self.next_terminated = t.zeros((envs.num_envs, self.num_agents), device=device, dtype=t.bool)  # need starting termination=False

    def play_step(self) -> list[dict]:
        """
        Carries out a single interaction step between the agent and the environment, and adds results to the replay memory.

        Returns the list of info dicts returned from `self.envs.step`.
        """
        # Get newest observations (i.e. where we're starting from)
        obs_global = self.next_obs
        if self.num_agents == 1:
            obs_global = obs_global.unsqueeze(dim=1)
        terminated = self.next_terminated

        # Compute logits based on newest observation, and use it to get an action distribution we sample from
        # TODO: add multiple agents, sample the action of each of them
        with t.inference_mode():
            logits = self.actor(obs_global)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)

        # Step environment based on the sampled action
        if self.num_agents == 1:
            input_actions = actions.squeeze(dim=1).cpu().numpy()
        else:
            input_actions = actions.cpu().numpy()
        next_obs_global, reward, next_terminated, next_truncated, infos = self.envs.step(input_actions)
        rewards = get_team_rewards(reward, infos, self.num_agents)
        assert rewards.shape == (self.envs.num_envs, self.num_agents)
        next_terminated = einops.repeat(next_terminated, "num_envs -> num_envs num_agents", num_agents=self.num_agents)
        next_truncated = einops.repeat(next_truncated, "num_envs -> num_envs num_agents", num_agents=self.num_agents)
        
        # Calculate logprobs and values, and add this all to replay memory
        # Vectorize code down below to compute everything for every agent at the same time (every shape should start with (num_envs, num_agents))
        environment_idx = t.arange(self.envs.num_envs, device=device)
        agent_idx = t.arange(self.num_agents, device=device)
        team_idx = agent_idx // self.team_size
        first_agent_idx = team_idx * self.team_size
        team_obs = obs_global[:, first_agent_idx, :]
        with t.inference_mode():
            values = self.critic(team_obs).squeeze(dim=-1).cpu().numpy()

        self.memory.add(
            obs = obs_global.cpu().numpy(),
            obs_global = team_obs.cpu().numpy(),
            actions = actions.cpu().numpy(),
            logprobs = logprobs.cpu().numpy(),
            values = values,
            rewards = rewards,
            terminated = terminated.cpu().numpy(),
        )
        """values = []
        for team_idx in range(self.num_teams):
            first_agent_idx = team_idx * self.team_size
            team_obs = obs_global[:, first_agent_idx]
            with t.inference_mode():
                values.append(self.critic(team_obs).flatten().cpu().numpy()) # TODO: .flatten needed?
        values = np.stack(values, axis=1)
        # Add to memory for each agent (obs, action, logprob seperately) (values, rewards, terminated together)
        for agent_idx in range(self.num_agents):
            team_idx = agent_idx // self.team_size
            actions = agent_actions[:, agent_idx]
            obs = obs_global[:, agent_idx]
            dist = agent_dists[agent_idx]
            logprobs = dist.log_prob(actions).cpu().numpy()
            self.memory.add(
                obs.cpu().numpy(),
                obs_global.cpu().numpy(),
                actions.cpu().numpy(),
                logprobs,
                values[:, team_idx],
                rewards[:, team_idx],
                terminated.cpu().numpy(),
                agent_idx,
            )"""

        # Set next observation & termination state
        self.next_obs = t.from_numpy(next_obs_global).to(device, dtype=t.float)
        self.next_terminated = t.from_numpy(next_terminated).to(device, dtype=t.float)

        self.step += self.envs.num_envs * self.num_agents
        return infos

    def get_minibatches(self, gamma: float, gae_lambda: float) -> list[ReplayMinibatch]:
        """
        Gets minibatches from the replay memory, and resets the memory
        """
        with t.inference_mode():
            next_value = self.critic(self.next_obs).squeeze(dim=-1)
        minibatches = self.memory.get_minibatches(next_value, self.next_terminated, gamma, gae_lambda)
        self.memory.reset()
        return minibatches
 

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
        self.num_teams = self.args.num_teams
        self.team_size = self.num_agents // self.num_teams

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
            self.num_agents,
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

        # Create our agents
        self.agents = PPOAgents(self.envs, self.actor, self.critic, self.memory, self.num_agents, self.num_teams)

    def rollout_phase(self) -> dict | None:
        """
        This function populates the memory with a new set of experiences, using `self.agents.play_step` to step through
        the environment. It also returns a dict of data which you can include in your progress bar postfix.
        """
        data = None
        t0 = time.time()

        for step in range(self.args.num_steps_per_rollout):
            # Play a step, returning the infos dict (containing information for each environment)
            infos = self.agents.play_step()

            # Get data from environments, and log it if some environment did actually terminate
            new_data = get_episode_data_from_infos(infos)
            if new_data is not None:
                data = new_data
                if self.args.use_wandb:
                    wandb.log(new_data, step=self.agents.step)

        if self.args.use_wandb:
            wandb.log(
                {"SPS": (self.args.num_steps_per_rollout * self.num_envs) / (time.time() - t0)}, step=self.agents.step
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
        minibatches = self.agents.get_minibatches(self.args.gamma, self.args.gae_lambda)
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
                    total_steps=self.agents.step,
                    values=values.mean().item(),
                    lr=self.scheduler.optimizer.param_groups[0]["lr"],
                    value_loss=value_loss.item(),
                    clipped_surrogate_objective=clipped_surrogate_objective.item(),
                    entropy=entropy_bonus.item(),
                    approx_kl=approx_kl,
                    clipfrac=np.mean(clipfracs),
                ),
                step=self.agents.step,
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
    agents = trainer.agents

    # Get the correct set of observations, and corresponding values we expect
    obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
    expected_value_for_probes = [[[1.0]], [[-1.0], [+1.0]], [[args.gamma], [1.0]], [[1.0]], [[1.0], [1.0]]]
    expected_probs_for_probes = [None, None, None, [[0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
    tolerances = [1e-3, 1e-3, 1e-3, 2e-3, 2e-3]
    obs = t.tensor(obs_for_probes[probe_idx - 1]).to(device)

    # Calculate the actual value & probs, and verify them
    with t.inference_mode():
        value = agents.critic(obs)
        probs = agents.actor(obs).softmax(-1)
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
        total_timesteps=35_000,
        use_wandb=False,
        gamma=0.0,
        num_envs=4,
    )
    trainer = PPOTrainer(args)
    trainer.train()
    agents = trainer.agents
    obs = t.tensor([[1.0, 0.0, 1.0, 0.0]]).to(device)
    with t.inference_mode():
        value = agents.critic(obs)
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
        num_teams=2,
        total_timesteps=35_000,
        use_wandb=False,
        gamma=0.0,
        num_envs=4,
    )
    trainer = PPOTrainer(args)
    trainer.train()
    agents = trainer.agents
    obs = t.tensor([[0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]]).to(device)
    with t.inference_mode():
        value = agents.critic(obs)
    print(value)
    expected_value = t.tensor([[1.0]]).to(device)
    t.testing.assert_close(value, expected_value, atol=5 * 1e-2, rtol=0)

# if MAIN:
#     for probe_idx in range(1, 6):
#         test_probe(probe_idx)

# if MAIN:
#     test_mappo()

if MAIN:
    test_mappo_selfplay()

# %%
if MAIN:
    args = PPOArgs(use_wandb=True, video_log_freq=50)
    trainer = PPOTrainer(args)
    trainer.train()


# %%
from gymnasium.envs.classic_control import CartPoleEnv


class EasyCart(CartPoleEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        x, v, theta, omega = obs

        # First reward: angle should be close to zero
        reward_1 = 1 - abs(theta / 0.2095)
        # Second reward: position should be close to the center
        reward_2 = 1 - abs(x / 2.4)

        # Combine both rewards (keep it in the [0, 1] range)
        reward_new = (reward_1 + reward_2) / 2

        return obs, reward_new, terminated, truncated, info


if MAIN:
    gym.envs.registration.register(id="EasyCart-v0", entry_point=EasyCart, max_episode_steps=500)
    args = PPOArgs(env_id="EasyCart-v0", use_wandb=True, video_log_freq=50)
    trainer = PPOTrainer(args)
    trainer.train()


# %%
class SpinCart(CartPoleEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        x, v, theta, omega = obs

        # Allow for 360-degree rotation (but keep the cart on-screen)
        terminated = abs(x) > self.x_threshold

        # Reward function incentivises fast spinning while staying still & near centre
        rotation_speed_reward = min(1, 0.1 * abs(omega))
        stability_penalty = max(1, abs(x / 2.5) + abs(v / 10))
        reward_new = rotation_speed_reward - 0.5 * stability_penalty

        return (obs, reward_new, terminated, truncated, info)

if MAIN:
    gym.envs.registration.register(id="SpinCart-v0", entry_point=SpinCart, max_episode_steps=500)
    args = PPOArgs(env_id="SpinCart-v0", use_wandb=True, video_log_freq=50)
    trainer = PPOTrainer(args)
    trainer.train()


# %%
if MAIN:
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

    print(env.action_space)  # Discrete(4): 4 actions to choose from
    print(env.observation_space)  # Box(0, 255, (210, 160, 3), uint8): an RGB image of the game screen


if MAIN:
    print(env.get_action_meanings())


# %%
def display_frames(frames: Int[Arr, "timesteps height width channels"], figsize=(4, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(frames[0])
    plt.close()

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=frames, interval=100)
    display(HTML(ani.to_jshtml()))


if MAIN:
    nsteps = 150

    frames = []
    obs, info = env.reset()
    for _ in tqdm(range(nsteps)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs)

    display_frames(np.stack(frames))


# %%
if MAIN:
    env_wrapped = prepare_atari_env(env)

    frames = []
    obs, info = env_wrapped.reset()
    for _ in tqdm(range(nsteps)):
        action = env_wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = env_wrapped.step(action)
        obs = einops.repeat(np.array(obs), "frames h w -> h (frames w) 3")  # stack frames across the row
        frames.append(obs)

    display_frames(np.stack(frames), figsize=(12, 3))


def get_actor_and_critic_atari(obs_shape: tuple[int,], num_actions: int) -> tuple[nn.Sequential, nn.Sequential]:
    """
    Returns (actor, critic) in the "atari" case, according to diagram above.
    """
    assert obs_shape[-1] % 8 == 4

    L_after_convolutions = (obs_shape[-1] // 8) - 3
    in_features = 64 * L_after_convolutions * L_after_convolutions

    hidden = nn.Sequential(
        layer_init(nn.Conv2d(4, 32, 8, stride=4, padding=0)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2, padding=0)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1, padding=0)),
        nn.ReLU(),
        nn.Flatten(),
        layer_init(nn.Linear(in_features, 512)),
        nn.ReLU(),
    )

    actor = nn.Sequential(hidden, layer_init(nn.Linear(512, num_actions), std=0.01))
    critic = nn.Sequential(hidden, layer_init(nn.Linear(512, 1), std=1))

    return actor, critic


# %%
if MAIN:
    args = PPOArgs(
        env_id="ALE/Breakout-v5",
        wandb_project_name="PPOAtari",
        use_wandb=True,
        mode="atari",
        clip_coef=0.1,
        num_envs=8,
        video_log_freq=25,
    )
    trainer = PPOTrainer(args)
    trainer.train()