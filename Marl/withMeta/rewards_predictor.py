from abc import ABC
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
from stable_baselines3.common.base_class import BaseAlgorithm

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import configure_logger, safe_mean
from stable_baselines3.common.policies import (ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy,
                                               MultiInputActorCriticPolicy)

sys.path.insert(1, "/home/acaftory/CommonsGame/DanfoaTest")
from Utils.trainer_utils import clip_action, correct_observation_indexing, correct_indexing, obs_as_tensor, map_dict
from Utils.data_base import TransitionBuffer


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


class TrainerRewardsPredictor(OnPolicyAlgorithm):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: GymEnv,
            learning_rate: Union[float, Schedule] = 1e-4,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            tensorboard_log: Optional[str] = None,
            verbose: int = 0,
            device: Union[th.device, str] = "auto",
            support_multi_env: bool = False,
            create_eval_env: bool = False,
            monitor_wrapper: bool = True,
            seed: Optional[int] = None,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,

            agent_observation_space: Box = None,
            agent_action_space: Discrete = None,
            num_agents: int = None,
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
            target_kl: Optional[float] = None,
            meta_policy: ActorCriticCnnPolicy = None,
            model_filename: str = "",
            alpha: float = 0.95,
            n_meta_action: int = 5,
    ):
        # init params
        self.env = env
        self.num_agents = num_agents
        self.n_envs = env.num_envs // (num_agents + 1)
        self.agents_observation_space = agent_observation_space
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

        dim = (env.observation_space.shape[-1],) + env.observation_space.shape[:-1]
        obs_space = gym.spaces.Box(low=0, high=255, shape=dim, dtype=np.uint8)
        meta_env_fn = lambda: DummyGymEnv(obs_space, self.meta_action_space)
        meta_dummy_env = DummyVecEnv([meta_env_fn] * self.n_envs)
        self.n_epochs = n_epochs
        super().__init__(
            policy=meta_policy,
            env=meta_dummy_env,
            learning_rate=learning_rate,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            policy_kwargs=policy_kwargs,
            supported_action_spaces=(
                gym.spaces.Box,
                gym.spaces.Discrete,
                gym.spaces.MultiDiscrete,
                gym.spaces.MultiBinary,
            ),
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            n_steps=n_steps,
            gamma=gamma,
        )
        self.env = env
        self.db = TransitionBuffer(max_size=int(5e4), batch_size=8, epsilon=0.005)
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
        self.meta = {self.num_agents: None}

        # copy rewards predictor weights from pretrained model
        self.policy.load_state_dict(th.load(model_filename))
        self.configure_loggers()

    def configure_loggers(self):
        for agent_id, agent in self.agents.items():
            agent._logger = configure_logger(0, None, "", True)

    def get_new_buffers(self):
        """
        Create new buffers for collect_rollouts
        :return: (actionBuffer, clipped_actionsBuffer, valuesBuffer, log_probs,
                 rewardsBuffer, observationsBuffer, donesBuffer, infosBuffer)
        """
        actions = []
        clipped_actions = []
        values = []
        log_probs = []
        rewards = []
        rewards_true = []
        observations = []
        dones = []
        infos = []

        return actions, clipped_actions, values, log_probs, rewards, rewards_true, observations, dones, infos

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
        for agent_id, agent in self.agents.items():
            agent.policy.set_training_mode(False)
            agent.rollout_buffer.reset()

        self._last_episode_starts = list(self._last_episode_starts.reshape(self.num_agents + 1, self.n_envs))
        observations = None
        values = None
        dones = None
        n_steps = 0
        rewards_per_r, rewards_per_fire = [], []
        transitions = np.zeros((self.n_envs, n_rollout_steps, *self.observation_space.shape))
        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            actions, clipped_actions, values, log_probs, rewards, rewards_true, observations, dones, infos = self.get_new_buffers()
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                [agent.policy.reset_noise(self.n_envs) for agent in self.agents]

            # predict actions by the agents

            # Convert to pytorch tensor or to TensorDict
            self.predict_low_level(self._last_obs, actions, values, log_probs, clipped_actions, observations,
                                   self.num_agents, self.n_envs, self.agents_observation_space)
            clipped_actions.append(np.zeros(self.n_envs, dtype=np.int64))
            all_clipped_actions = np.vstack(clipped_actions).transpose().reshape(-1)
            new_obs, rewards_t, dones_, infos_ = env.step(all_clipped_actions)

            # meta-agent reward
            meta_obs = correct_observation_indexing(new_obs, self.num_agents + 1, 2, self.n_envs,
                                                    self.meta_observation_space.shape).transpose(0, -1, 1, 2)
            with th.no_grad():
                meta_obs = obs_as_tensor(meta_obs, self.policy.device)
            rewards_, _, _ = self.policy.forward(meta_obs)
            rewards_ = np.array([self.meta_rewards[i] for i in rewards_.cpu().numpy().ravel()])

            # reshape rewards/done/info to fit the indexing convention
            for agent_id, agent in self.agents.items():
                rewards.append(correct_indexing(rewards_, self.num_agents, agent_id, self.n_envs))
                rewards_true.append(correct_indexing(rewards_t, self.num_agents + 1, agent_id, self.n_envs))
                dones.append(correct_indexing(dones_, self.num_agents, agent_id, self.n_envs))
                infos.append(list(correct_indexing(infos_, self.num_agents + 1, agent_id, self.n_envs)))

            self.num_timesteps += self.n_envs

            # ugly hack to make the indexing convention work
            if not np.any(dones):
                fire_indices = [x[1] for x in sorted([(i, agent_info["fire"]) for inf in infos
                                                      for i, agent_info in enumerate(inf)],
                                                     key=lambda x: x[0])]
                reward_per_fire = safe_mean(rewards_[fire_indices])
                rewards_per_fire.append(reward_per_fire)
            rewards_per_r.append(safe_mean(np.array(rewards)[np.where(rewards_true)]))

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            # store transitions in agents buffers
            for agent_id, agent in self.agents.items():
                agent.rollout_buffer.add(
                    observations[agent_id],
                    np.expand_dims(actions[agent_id], 1),
                    rewards[agent_id],
                    self._last_episode_starts[agent_id],
                    values[agent_id],
                    log_probs[agent_id])

            #  store transition for rewards predictor
            for i in range(self.n_envs):
                transitions[i, n_steps, :] = meta_obs[i].detach().cpu()
            n_steps += 1
            self._last_obs = new_obs
            self._last_episode_starts = dones
        # Compute value for the last timestep
        with th.no_grad():
            for agent_id, agent in self.agents.items():
                if (agent_id + 1) % (self.num_agents + 1) != 0:
                    _obs = correct_observation_indexing(self._last_obs, self.num_agents + 1, agent_id, self.n_envs,
                                                        self.agents_observation_space.shape).transpose(0, -1, 1, 2)
                obs_tensor = obs_as_tensor(_obs, agent.policy.device)

                # Compute value for the last timestep
                values[agent_id] = agent.policy.predict_values(obs_as_tensor(_obs, self.device))
                agent.rollout_buffer.compute_returns_and_advantage(last_values=values[agent_id], dones=dones[agent_id])
        # store transitions in meta-agent buffer
        self.db.add(transitions, infos)

        # update meta agent infos
        meta_info = correct_indexing(infos_, self.num_agents + 1, self.num_agents, self.n_envs)
        self._update_info_buffer(meta_info, rewards_per_r, rewards_per_fire)
        callback.on_rollout_end()
        self._last_episode_starts = np.ones(((self.num_agents + 1) * self.n_envs, 1))
        return True

    def meta_agent_train(self):
        self.policy.set_training_mode(True)
        db = self.db.sample()
        losses = []
        ps = []
        for _ in range(self.n_epochs):
            loss = th.tensor(0, dtype=th.float).to(self.device)
            rewards = th.Tensor(self.meta_rewards).to(self.device)
            rewards_s = th.zeros(self.n_steps)
            for record in db:
                obs1 = obs_as_tensor(record.segment1.transitions, self.device)
                obs2 = obs_as_tensor(record.segment2.transitions, self.device)
                r1 = self.policy.get_actions(obs1, record.segment1.agent_id, rewards)
                r2 = self.policy.get_actions(obs2, record.segment1.agent_id, rewards)

                # p = th.exp(r1.mean()) / (th.exp(r1.mean()) + th.exp(r2.mean()))
                p = r1.sum() / (r2.sum() + r1.sum())
                ps.append(max([p.item(), 1 - p.item()]))
                loss -= record.mu[0] * th.log(p) + record.mu[1] * th.log(1 - p)

            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()
            th.cuda.empty_cache()
            losses.append(loss.item())
        return np.mean(losses), np.mean(ps)

    def train(self) -> None:
        """
        Update agents policies using the currently gathered rollout buffer.
       """
        episode_logg = []
        for agent_id, agent in self.agents.items():
            # for agent_id, agent in self.agents.items():
            agent.train()
            episode_logg.append(agent._logger.name_to_value)
        episode_logg = map_dict(episode_logg)

        if len(self.db) > self.db.batch_size:
            loss, ps = self.meta_agent_train()
            episode_logg["Meta/Loss"] = [loss]
            episode_logg["Meta/Probabilities"] = [ps]

        self.logg(episode_logg)

    def logg(self, episode_logg: Dict):
        """
        Collect statistics from learning and export it to an internal logger
        :param episode_logg: Dictionary of <Tag (str): statistic values (List)>
        """
        for k, v in episode_logg.items():
            self.logger.record(k, safe_mean(v))

        # agents metrics
        self.logger.record("metrics/efficiency", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
        self.logger.record("metrics/equality", safe_mean([ep_info["equality"] for ep_info in self.ep_info_buffer]))
        self.logger.record("metrics/sustainability",
                           safe_mean([ep_info["sustainability"] for ep_info in self.ep_info_buffer]))
        self.logger.record("metrics/peace", safe_mean([ep_info["peace"] for ep_info in self.ep_info_buffer]))
        self.logger.record("Meta/Predicted_per_Apple", safe_mean([ep_info["preds_per_r"]
                                                                  for ep_info in self.ep_info_buffer]))
        self.logger.record("Meta/Predicted_per_Fire", safe_mean([ep_info["preds_per_fire"]
                                                                 for ep_info in self.ep_info_buffer]))


    def predict_low_level(self, observations, actions, values, log_probs, clipped_actions, observations_,
                          num_agents, n_envs, agents_observation_space):
        with th.no_grad():
            for agent_id, agent in self.agents.items():
                # checks rather current agent is meta or regular
                if (agent_id + 1) % (self.num_agents + 1):  # low-level agent
                    _obs = correct_observation_indexing(observations, num_agents + 1, agent_id, n_envs,
                                                        agents_observation_space.shape).transpose(0, -1, 1, 2)
                    observations_.append(_obs)
                    obs_tensor = obs_as_tensor(_obs, agent.policy.device)

                    #  feed forward agents observations
                    actions_, values_, log_probs_ = agent.policy.forward(obs_tensor)

                    # store actions, values, log_probs
                    actions.append(actions_.cpu().numpy())
                    values.append(values_)
                    log_probs.append(log_probs_)
                    clipped_actions.append(clip_action(actions[agent_id], self.action_space))

    def learn(self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 1,
              eval_env: Optional[GymEnv] = None, eval_freq: int = -1, n_eval_episodes: int = 5,
              tb_log_name: str = "Trail", eval_log_path: Optional[str] = None, reset_num_timesteps: bool = True):

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

    def _update_info_buffer(self, infos, preds_for_apples, preds_per_fire) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        for idx, info in enumerate(infos):
            dic = {"preds_per_r": np.nanmean(preds_for_apples), "preds_per_fire": np.nanmean(preds_per_fire)}
            self.ep_info_buffer.extend([info.get("metrics") | dic])
