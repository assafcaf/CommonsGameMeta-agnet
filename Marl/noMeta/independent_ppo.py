from typing import Any, Dict, List, Optional, Type, Union, Tuple
import numpy as np
import gym
import os
from .utils import *
import torch as th
from gym.core import ActType, ObsType, RenderFrame
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import configure_logger, safe_mean
from stable_baselines3.common.policies import (ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy,
                                               MultiInputActorCriticPolicy)


class DummyGymEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def step(self, action: ActType) -> Union[
        Tuple[ObsType, float, bool, bool, dict], Tuple[ObsType, float, bool, dict]
    ]:
        pass


class IndependentPPO(OnPolicyAlgorithm):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            num_agents: int,
            env: GymEnv,
            learning_rate: Union[float, Schedule] = 1e-4,
            n_steps: int = 1000,
            batch_size: int = 6000,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 1.0,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 40,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            create_eval_env: bool = False,
            device: Union[th.device, str] = "auto"):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                gym.spaces.Box,
                gym.spaces.Discrete,
                gym.spaces.MultiDiscrete,
                gym.spaces.MultiBinary,
            ),
        )

        # init params
        self.env = env
        self.num_agents = num_agents
        self.n_envs = env.num_envs // num_agents
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_steps = n_steps
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self._logger = None

        env_fn = lambda: DummyGymEnv(self.observation_space, self.action_space)
        dummy_env = DummyVecEnv([env_fn] * self.n_envs)

        self.agents = [PPO(
            policy=policy,
            env=dummy_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device)
            for _ in range(self.num_agents)]
        self.configure_loggers()

    def configure_loggers(self):
        for agent in self.agents:
            agent._logger = configure_logger(0, None, "", True)

    def get_new_buffers(self):
        """
        Create new buffers for collect_rollouts
        :return: (actionBuffer, clipped_actionsBuffer, valuesBuffer, log_probs,
                 rewardsBuffer, observationsBuffer, donesBuffer, infosBuffer)
        """
        actions = [None] * self.num_agents
        clipped_actions = [None] * self.num_agents
        values = [None] * self.num_agents
        log_probs = [None] * self.num_agents
        rewards = [None] * self.num_agents
        observations = np.zeros((self.num_agents, self.n_envs) + self.observation_space.shape[::-1], dtype=np.uint8)
        dones = [None] * self.num_agents
        infos = [None] * self.num_agents

        return actions, clipped_actions, values, log_probs, rewards, observations, dones, infos

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer,
                         n_rollout_steps: int) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        # config agents to collect rollouts mode
        for agent in self.agents:
            agent.policy.set_training_mode(False)
            agent.rollout_buffer.reset()

        self._last_episode_starts = list(self._last_episode_starts.reshape(self.num_agents, self.n_envs))
        observations = None
        values = None
        dones = None
        n_steps = 0
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            actions, clipped_actions, values, log_probs, rewards, observations,dones, infos = self.get_new_buffers()
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                [agent.policy.reset_noise(self.n_envs) for agent in self.agents]

            # predict actions by the agents
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                for agent_id, agent in enumerate(self.agents):
                    observations[agent_id, :] = correct_indexing(self._last_obs, self.num_agents,
                                                                 agent_id, self.n_envs).transpose((0, -1, 1, 2))

                    obs_tensor = obs_as_tensor(observations[agent_id, :], agent.policy.device)
                    actions_, values_, log_probs_ = agent.policy.forward(obs_tensor)

                    actions[agent_id] = np.expand_dims(actions_.cpu().numpy(), -1)
                    values[agent_id] = values_
                    log_probs[agent_id] = log_probs_
                    clipped_actions[agent_id] = clip_action(actions_, self.action_space)

            all_clipped_actions = np.vstack(clipped_actions).transpose().reshape(-1)
            new_obs, rewards_, dones_, infos_ = env.step(all_clipped_actions)

            # reshape rewards/done/info to fit the indexing convention
            for agent_id in range(self.num_agents):
                rewards[agent_id] = correct_indexing(rewards_, self.num_agents, agent_id, self.n_envs)
                dones[agent_id] = correct_indexing(dones_, self.num_agents, agent_id, self.n_envs)
                infos[agent_id] = correct_indexing(infos_, self.num_agents, agent_id, self.n_envs)


            self.num_timesteps += self.n_envs
            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False
            if n_steps + 1 == n_rollout_steps:
                stop = True
            [self._update_info_buffer(infos[i]) for i, agent in enumerate(self.agents)]
            n_steps += 1

            # store transitions in agents buffers
            for agent_id, agent in enumerate(self.agents):
                agent.rollout_buffer.add(
                    observations[agent_id, :],
                    actions[agent_id],
                    rewards[agent_id],
                    self._last_episode_starts[agent_id],
                    values[agent_id],
                    log_probs[agent_id])
            self._last_obs = new_obs
            self._last_episode_starts = dones

        # Compute value for the last timestep
        with th.no_grad():
            for agent_id, agent in enumerate(self.agents):
                observations[agent_id, :] = correct_indexing(self._last_obs, self.num_agents,
                                                             agent_id, self.n_envs).transpose((0, -1, 1, 2))

                # Compute value for the last timestep
                values[agent_id] = agent.policy.predict_values(obs_as_tensor(observations[agent_id, :], self.device))
                agent.rollout_buffer.compute_returns_and_advantage(last_values=values[agent_id], dones=dones[agent_id])
        callback.on_rollout_end()
        self._last_episode_starts = np.ones((self.num_agents * self.n_envs, 1))
        return True

    def train(self) -> None:
        """
        Update agents policies using the currently gathered rollout buffer.
       """
        episode_logg = []
        for agent in self.agents:
            agent.train()
            episode_logg.append(agent._logger.name_to_value)

        episode_logg = map_dict(episode_logg)
        self.logg(episode_logg)

    def logg(self, episode_logg: Dict):
        """
        Collect statistics from learning and export it to an internal logger
        :param episode_logg: Dictionary of <Tag (str): statistic values (List)>
        """
        for k, v in episode_logg.items():
            self.logger.record(k, safe_mean(v))
        self.logger.record("metrics/efficiency", safe_mean([ep_info["efficiency"] for ep_info in self.ep_info_buffer]))
        self.logger.record("metrics/equality", safe_mean([ep_info["equality"] for ep_info in self.ep_info_buffer]))
        self.logger.record("metrics/sustainability", safe_mean([ep_info["sustainability"] for ep_info in self.ep_info_buffer]))
        self.logger.record("metrics/peace", safe_mean([ep_info["peace"] for ep_info in self.ep_info_buffer]))

    def predict_(self, observation: np.ndarray,  n_envs: Optional[np.ndarray] = None, deterministic: bool = False,)\
            -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        observations_ = np.zeros((self.num_agents, n_envs) + self.observation_space.shape[::-1], dtype=np.uint8)
        clipped_actions = [None] * self.num_agents
        for agent_id, agent in enumerate(self.agents):
            observations_[agent_id, :] = correct_indexing(observation, self.num_agents,
                                                          agent_id, n_envs).transpose((0, -1, 1, 2))

            obs_tensor = obs_as_tensor(observations_[agent_id, :], agent.policy.device)
            actions_, _, _ = agent.policy.forward(obs_tensor, deterministic=deterministic)
            clipped_actions[agent_id] = clip_action(actions_, self.action_space)

        return np.vstack(clipped_actions).transpose().reshape(-1)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "IndependentPPO",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            maybe_ep_metrics = info.get("metrics")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info | maybe_ep_metrics])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)