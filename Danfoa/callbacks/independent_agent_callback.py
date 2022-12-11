from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from PIL import Image as im
import imageio
import os
import json


class IndependentAgentCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, eval_env, verbose=0, freq=1000):
        super(IndependentAgentCallback, self).__init__(verbose)
        self.iterations_ = 0
        self.freq = freq
        self.eval_env = eval_env

    def _on_training_start(self) -> None:
        file_name = os.path.join(self.model.logger.dir, "parameters.json")

        params = {"learning_rate": self.model.learning_rate,
                  "batch_size": self.model.agents[0].batch_size,
                  "gae_lambda": self.model.gae_lambda,
                  "gamma": self.model.gamma,
                  "n_envs": self.model.n_envs,
                  "n_epochs": self.model.agents[0].n_epochs,
                  "normalize_advantage": self.model.agents[0].normalize_advantage,
                  "target_kl": self.model.agents[0].target_kl,
                  "ent_coef": self.model.ent_coef,
                  "n_frames": self.model.env.observation_space.shape[-1],
                  "policy_type": str(type(self.model.agents[0].policy.features_extractor)).split(".")[-1],
                  "observations_space": str(self.model.observation_space),
                  "k": self.model.k,
                  "actions": str(self.eval_env.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env.spawn_prob)
                  }

        json_object = json.dumps(params, indent=4)
        with open(file_name, "w") as outfile:
            outfile.write(json_object)

    def _on_step(self) -> bool:
        pass

    def _on_rollout_end(self) -> None:
        if (self.iterations_ + 1) % self.freq == 0:
            play_env = self.eval_env
            render_env = self.eval_env.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env
            observations = play_env.reset()
            frames = []
            score = 0
            for _ in range(1000):
                actions = self.model.predict_(observations, 1)
                observations, rewards, dones, infos = play_env.step(actions.astype(np.uint8))
                frame = render_env.render(mode="RGB")
                frames.append(
                    im.fromarray(frame.astype(np.uint8)).resize(size=(720, 480), resample=im.BOX).convert("RGB"))
                score += rewards.sum()

            file_name = self.logger.dir + f"/iteration_{self.iterations_ + 1}_score_{int(score)}.gif"
            imageio.mimsave(file_name, frames, fps=15)

        self.iterations_ += 1

