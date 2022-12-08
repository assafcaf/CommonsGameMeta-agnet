from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from PIL import Image as im
import imageio
import os
import json


class SingleAgentCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, eval_env, verbose=0, freq=1000):
        super(SingleAgentCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.iterations_ = 0
        self.freq = freq
        self.eval_env = eval_env

    def _on_training_start(self) -> None:
        file_name = os.path.join(self.model.logger.dir, "parameters.json")

        params = {"learning_rate": self.model.learning_rate,
                  "batch_size": self.model.batch_size,
                  "gae_lambda": self.model.gae_lambda,
                  "gamma": self.model.gamma,
                  "n_envs": self.model.n_envs,
                  "n_epochs": self.model.n_epochs,
                  "normalize_advantage": self.model.normalize_advantage,
                  "target_kl": self.model.target_kl,
                  "ent_coef": self.model.ent_coef,
                  # each frame consist from 3 channels (RGB)
                  "n_frames": self.model.env.observation_space.shape[0],
                  "policy_kwargs": self.model.policy.features_extractor_kwargs,
                  "policy_type": str(type(self.model.policy.features_extractor)).split(".")[-1],
                  "observations_space": str(self.model.observation_space)}

        json_object = json.dumps(params, indent=4)
        with open(file_name, "w") as outfile:
            outfile.write(json_object)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
            This method will be called by the model after each call to `env.step()`.

            For child callback (of an `EventCallback`), this will be called
            when the event is triggered.

            :return: (bool) If the callback returns False, training is aborted early.
            """

        if self.n_calls % self.freq == 0:
            # env = self.model.env.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env
            play_env = self.eval_env
            render_env = self.eval_env.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env
            observations = play_env.reset()
            frames = []
            score = 0
            for _ in range(1000):
                # TODO
                actions, _ = self.model.predict(observations, state=None, deterministic=False)
                observations, rewards, dones, infos = play_env.step(actions.astype(np.uint8))
                frame = render_env.render(mode="RGB")
                frames.append(im.fromarray(frame.astype(np.uint8)).resize(size=(720, 480), resample=im.BOX).convert("RGB"))
                score += rewards.sum()

            file_name = self.logger.dir + f"/iteration_{self.iterations_+1}_score_{int(score)}.gif"
            imageio.mimsave(file_name, frames, fps=15)
        return True

    def _play(self):
        observations = self.eval_env.reset()
        frames = []
        score = 0
        values = 0
        for t in range(1000):
            actions, states = self.model.predict(observations, state=None, deterministic=False)
            observations, rewards, dones, infos = self.eval_env.step(actions.astype(np.uint8))

    def _on_rollout_end(self) -> None:
        self.iterations_ += 1
        self._play()
        soc_met = self.eval_env.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env.get_social_metrics()
        ef, eq, sus, p = soc_met
        self.logger.record("metrics/efficiency", ef)
        self.logger.record("metrics/equality", eq)
        self.logger.record("metrics/sustainability", sus)
        self.logger.record("metrics/peace", p)
        # self.logger.record("metrics/sum_of_rewards", sor)
        # self.logger.record("metrics/shot_accuracy", sac)
        self.logger.record("metrics/episodes", self.iterations_)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class IndependentAgentCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, eval_env, verbose=0, freq=1000):
        super(IndependentAgentCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.iterations_ = 0
        self.freq = freq
        self.eval_env = eval_env

    def _on_training_start(self) -> None:
        file_name = os.path.join(self.model.logger.dir, "parameters.json")

        params = {"learning_rate": self.model.learning_rate,
                  "batch_size": self.model.batch_size,
                  "gae_lambda": self.model.gae_lambda,
                  "gamma": self.model.gamma,
                  "n_envs": self.model.n_envs,
                  "n_epochs": self.model.n_epochs,
                  "normalize_advantage": self.model.normalize_advantage,
                  "target_kl": self.model.target_kl,
                  "ent_coef": self.model.ent_coef,
                  # each frame consist from 3 channels (RGB)
                  "n_frames": self.model.env.observation_space.shape[0],
                  "policy_kwargs": self.model.policies[0].features_extractor_kwargs,
                  "policy_type": str(type(self.model.policies[0].features_extractor)).split(".")[-1],
                  "observations_space": str(self.model.observation_space)}

        json_object = json.dumps(params, indent=4)
        with open(file_name, "w") as outfile:
            outfile.write(json_object)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
            This method will be called by the model after each call to `env.step()`.

            For child callback (of an `EventCallback`), this will be called
            when the event is triggered.

            :return: (bool) If the callback returns False, training is aborted early.
            """

        if self.n_calls % self.freq == 0:
            # env = self.model.env.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env
            play_env = self.eval_env
            render_env = self.eval_env.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env
            observations = play_env.reset()
            frames = []
            score = 0
            for _ in range(1000):
                # TODO
                actions, _ = self.model.predict(observations, state=None, deterministic=False)
                observations, rewards, dones, infos = play_env.step(actions.astype(np.uint8))
                frame = render_env.render(mode="RGB")
                frames.append(im.fromarray(frame.astype(np.uint8)).resize(size=(720, 480), resample=im.BOX).convert("RGB"))
                score += rewards.sum()

            file_name = self.logger.dir + f"/iteration_{self.iterations_+1}_score_{int(score)}.gif"
            imageio.mimsave(file_name, frames, fps=15)
        return True

    def _play(self):
        env = self.eval_env
        observations = env.reset()
        frames = []
        score = 0
        values = 0
        for _ in range(1000):
            actions, states = self.model.predict(observations, state=None, deterministic=False)
            observations, rewards, dones, infos = env.step(actions.astype(np.uint8))

    def _on_rollout_end(self) -> None:
        self.iterations_ += 1
        avg_values = self._play()
        efficiency, equality, sustainability, peace = [], [], [], []
        soc_met = self.eval_env.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env.get_social_metrics()
        ef, eq, sus, p, sor, sac = soc_met
        self.logger.record("metrics/efficiency", ef)
        self.logger.record("metrics/equality", eq)
        self.logger.record("metrics/sustainability", sus)
        self.logger.record("metrics/peace", p)
        self.logger.record("metrics/sum_of_rewards", sor)
        self.logger.record("metrics/shot_accuracy", sac)
        self.logger.record("metrics/Values", avg_values)
        self.logger.record("metrics/episodes", self.iterations_)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class CustomIndependentCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, eval_env, verbose=0, freq=1000):
        super(CustomIndependentCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.iterations_ = 0
        self.freq = freq
        self.eval_env = eval_env

    def set_model(self, model):
        self.model = model
        return self

    def _on_training_start(self) -> None:
        file_name = os.path.join(self.model.logger.dir, "parameters.json")

        params = {"learning_rate": self.model.learning_rate,
                  "batch_size": self.model.batch_size,
                  "gae_lambda": self.model.gae_lambda,
                  "gamma": self.model.gamma,
                  "n_envs": self.model.n_envs,
                  "n_epochs": self.model.n_epochs,
                  "normalize_advantage": self.model.normalize_advantage,
                  "target_kl": self.model.target_kl,
                  "ent_coef": self.model.ent_coef}

        json_object = json.dumps(params, indent=4)
        with open(file_name, "w") as outfile:
            outfile.write(json_object)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
            This method will be called by the model after each call to `env.step()`.

            For child callback (of an `EventCallback`), this will be called
            when the event is triggered.

            :return: (bool) If the callback returns False, training is aborted early.
            """

        if self.n_calls % self.freq == 0:
            # env = self.model.env.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env
            play_env = self.eval_env
            render_env = self.eval_env.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env
            observations = play_env.reset()
            frames = []
            score = 0
            for _ in range(1000):
                # TODO
                actions, states = self.model.predict(observations, state=None, deterministic=False)
                observations, rewards, dones, infos = play_env.step(actions.astype(np.uint8))
                frame = render_env.render(mode="RGB")
                frames.append(im.fromarray(frame.astype(np.uint8)).resize(size=(720, 480), resample=im.BOX).convert("RGB"))
                score += rewards.sum()

            file_name = self.logger.dir + f"/iteration_{self.iterations_+1}_score_{int(score)}.gif"
            imageio.mimsave(file_name, frames, fps=15)
        return True

    def _play(self):
        env = self.eval_env
        observations = env.reset()
        frames = []
        score = 0
        for _ in range(1000):
            actions, states = self.model.predict(observations, state=None, deterministic=False)
            observations, rewards, dones, infos = env.step(actions.astype(np.uint8))

    def _on_rollout_end(self) -> None:
        self.iterations_ += 1
        # self._play()
        efficiency, equality, sustainability, peace = [], [], [], []
        ef, eq, sus, p, sor, sac = self.eval_env.par_env.env.aec_env.env.env.env.ssd_env.get_social_metrics()
        self.logger.record("metrics/efficiency", ef)
        self.logger.record("metrics/equality", eq)
        self.logger.record("metrics/sustainability", sus)
        self.logger.record("metrics/peace", p)
        self.logger.record("metrics/sum_of_rewards", sor)
        self.logger.record("metrics/shot_accuracy", sac)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass