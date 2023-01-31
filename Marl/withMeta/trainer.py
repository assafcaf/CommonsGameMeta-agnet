from typing import Any, Dict, List, Optional, Type, Union, Tuple
import gym
import sys
import numpy as np
from gym.spaces import MultiDiscrete, Discrete, Box
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
sys.path.insert(1, "/home/acaftory/CommonsGame/DanfoaTest")
from Utils.trainer_utils import clip_action, correct_observation_indexing, correct_indexing, obs_as_tensor, map_dict


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


class TrainerWithMeta(OnPolicyAlgorithm):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            agent_observation_space: Box,
            agent_action_space: Discrete,
            num_agents: int,
            env: GymEnv,
            k: int = 25,
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
            meta_policy: ActorCriticCnnPolicy = None,
            model_filename: str = "",
            alpha: float = 0.95,
            n_meta_action: int = 5,
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
        self.n_envs = env.num_envs // (num_agents + 1)
        self.agents_observation_space = agent_observation_space
        self.k = k
        self.alpha = alpha

        self.meta_observation_space = env.observation_space
        self.meta_action_space = gym.spaces.MultiDiscrete([n_meta_action] * self.num_agents)
        self.meta_rewards = np.linspace(-1, 1, n_meta_action)

        self.n_steps = n_steps
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self._logger = None


        agents_env_fn = lambda: DummyGymEnv(self.agents_observation_space, agent_action_space)
        agents_dummy_env = DummyVecEnv([agents_env_fn] * self.n_envs)

        dim = (env.observation_space.shape[-1], ) + env.observation_space.shape[:-1]
        obs_space = gym.spaces.Box(low=0, high=255, shape=dim, dtype=np.uint8)
        meta_env_fn = lambda: DummyGymEnv(obs_space, self.meta_action_space)
        meta_dummy_env = DummyVecEnv([meta_env_fn] * self.n_envs)

        self.agents = {i: PPO(
            policy=policy,
            env=agents_dummy_env,
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
            policy_kwargs=None,
            verbose=0,
            device=device)
            for i in range(self.num_agents)}

        meta_ppo = PPO(
            policy=meta_policy,
            env=meta_dummy_env,
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
        meta_ppo.policy.load_state_dict(th.load(model_filename))
        self.meta = {len(self.agents): meta_ppo}

        self.configure_loggers()

    def configure_loggers(self):
        for agent_id, agent in (self.agents | self.meta).items():
            agent._logger = configure_logger(0, None, "", True)

    def get_new_buffers(self):
        """
        Create new buffers for collect_rollouts
        :return: (actionBuffer, clipped_actionsBuffer, valuesBuffer, log_probs,
                 rewardsBuffer, observationsBuffer, donesBuffer, infosBuffer)
        """
        actions = [None] * (self.num_agents + 1)
        clipped_actions = [None] * (self.num_agents + 1)
        values = [None] * (self.num_agents + 1)
        log_probs = [None] * (self.num_agents + 1)
        rewards = [None] * (self.num_agents + 1)
        # agents_observations = np.zeros((self.num_agents, self.n_envs) + self.agents_observation_space.shape[::-1],
        #                                dtype=np.uint8)
        # meta_observations = np.zeros((1, self.n_envs) + self.observation_space.shape, dtype=np.uint8)
        observations = [None] * (self.num_agents + 1)
        dones = [None] * (self.num_agents + 1)
        infos = [None] * (self.num_agents + 1)

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
        for agent_id, agent in (self.agents | self.meta).items():
            agent.policy.set_training_mode(False)
            agent.rollout_buffer.reset()

        self._last_episode_starts = list(self._last_episode_starts.reshape(self.num_agents + 1, self.n_envs))
        n_steps = 0
        values = None
        dones = None
        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            actions, clipped_actions, values, log_probs, rewards, observations, dones, infos = self.get_new_buffers()
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                [agent.policy.reset_noise(self.n_envs) for agent in self.agents]

            # predict actions by the agents
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                for agent_id, agent in self.agents.items():
                    # checks rather current agent is meta or regular
                    _obs = correct_observation_indexing(self._last_obs, self.num_agents+1, agent_id, self.n_envs,
                                                        self.agents_observation_space.shape).transpose(0, -1, 1, 2)
                    observations[agent_id] = _obs
                    obs_tensor = obs_as_tensor(_obs, agent.policy.device)

                    #  feed forward agents observations
                    actions_, values_, log_probs_ = agent.policy.forward(obs_tensor)
                    actions[agent_id] = np.expand_dims(actions_.cpu().numpy(), -1)
                    values[agent_id] = values_
                    log_probs[agent_id] = log_probs_
                    clipped_actions[agent_id] = clip_action(actions_, self.action_space)

            # tale one step in the environment
            clipped_actions[-1] = np.zeros(self.n_envs).astype(int)-1
            all_clipped_actions = np.vstack(clipped_actions).transpose().reshape(-1)
            new_obs, rewards_, dones_, infos_ = env.step(all_clipped_actions)

            # meta-agent rewards
            with th.no_grad():
                meta_obs = correct_observation_indexing(new_obs, self.num_agents+1, 2, self.n_envs,
                                                                self.meta_observation_space.shape).transpose(0, -1, 1, 2)
                meta_obs_t = obs_as_tensor(meta_obs, self.meta[2].policy.device)
                meta_actions, meta_values, meta_log_prob = self.meta[self.num_agents].policy.forward(meta_obs_t)
                meta_actions = meta_actions.cpu().numpy()
                rewards_m = np.array([self.meta_rewards[i] for i in meta_actions.ravel()])

            # reshape rewards/done/info to fit the indexing convention
            for agent_id, agent in (self.agents | self.meta).items():
                if agent_id < self.num_agents:
                    rewards[agent_id] = correct_indexing(rewards_m, self.num_agents, agent_id, self.n_envs)
                    dones[agent_id] = correct_indexing(dones_, self.num_agents, agent_id, self.n_envs)
                    infos[agent_id] = correct_indexing(infos_, self.num_agents, agent_id, self.n_envs)
                else:
                    rewards[agent_id] = correct_indexing(rewards_, self.num_agents, agent_id, self.n_envs)
                    dones[agent_id] = correct_indexing(dones_, self.num_agents, agent_id, self.n_envs)
                    infos[agent_id] = correct_indexing(infos_, self.num_agents, agent_id, self.n_envs)

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False
            [self._update_info_buffer(infos[agent_id]) for agent_id, agent in self.agents.items()]

            # store transitions in agents buffers
            for agent_id, agent in (self.agents | self.meta).items():
                if agent_id == self.num_agents:
                    agent.rollout_buffer.add(
                        meta_obs,
                        meta_actions,
                        rewards[agent_id],
                        self._last_episode_starts[agent_id],
                        meta_values,
                        meta_log_prob)
                else:
                    agent.rollout_buffer.add(
                        observations[agent_id],
                        actions[agent_id],
                        rewards[agent_id],
                        self._last_episode_starts[agent_id],
                        values[agent_id],
                        log_probs[agent_id])

            # end of transition storing
            self.num_timesteps += self.n_envs
            n_steps += 1
            self._last_obs = new_obs
            self._last_episode_starts = dones

        # Compute value for the last timestep
        with th.no_grad():
            for agent_id, agent in (self.agents | self.meta).items():
                if (agent_id+1) % (self.num_agents+1) != 0:  # if not meta-agent
                    _obs = correct_observation_indexing(self._last_obs, self.num_agents + 1, agent_id, self.n_envs,
                                                        self.agents_observation_space.shape).transpose(0, -1, 1, 2)
                else:  # meta-agent
                    _obs = correct_observation_indexing(self._last_obs, self.num_agents + 1, agent_id, self.n_envs,
                                                        self.meta_observation_space.shape).transpose(0, -1, 1, 2)
                obs_tensor = obs_as_tensor(_obs, agent.policy.device)

                # Compute value for the last timestep
                values[agent_id] = agent.policy.predict_values(obs_as_tensor(_obs, self.device))
                agent.rollout_buffer.compute_returns_and_advantage(last_values=values[agent_id], dones=dones[agent_id])
        callback.on_rollout_end()
        self._last_episode_starts = np.ones(((self.num_agents+1) * self.n_envs, 1))
        return True

    def train(self) -> None:
        """
        Update agents policies using the currently gathered rollout buffer.
       """
        episode_logg = []
        for agent_id, agent in (self.agents | self.meta).items():
        # for agent_id, agent in self.agents.items():
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

        # agents metrics
        self.logger.record("metrics/efficiency", safe_mean([ep_info["efficiency"] for ep_info in self.ep_info_buffer]))
        self.logger.record("metrics/equality", safe_mean([ep_info["equality"] for ep_info in self.ep_info_buffer]))
        self.logger.record("metrics/sustainability",
                           safe_mean([ep_info["sustainability"] for ep_info in self.ep_info_buffer]))
        self.logger.record("metrics/peace", safe_mean([ep_info["peace"] for ep_info in self.ep_info_buffer]))

    def predict_(self, observation: np.ndarray, n_envs: Optional[np.ndarray] = None, deterministic: bool = False, ) \
            -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        clipped_actions = [None] * (self.num_agents + 1)
        for agent_id, agent in self.agents.items():
            # checks rather current agent is meta or regular
            if (agent_id+1) % (self.num_agents+1):
                _obs = correct_observation_indexing(observation, self.num_agents + 1, agent_id, 1,
                                                    self.agents_observation_space.shape).transpose(0, -1, 1, 2)
            else:
                _obs = correct_observation_indexing(self._last_obs, self.num_agents + 1, agent_id, 1,
                                                    self.meta_observation_space.shape).transpose(0, -1, 1, 2)

            obs_tensor = obs_as_tensor(_obs, agent.policy.device)
            actions_, _, _ = agent.policy.forward(obs_tensor, deterministic=deterministic)
            clipped_actions[agent_id] = clip_action(actions_, self.action_space)
        clipped_actions[-1] = np.zeros(1).astype(int) - 1
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
