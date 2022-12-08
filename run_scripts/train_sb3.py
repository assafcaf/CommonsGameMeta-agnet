import sys
import os
import gym
import copy
import torch
import argparse
import supersuit as ss
from typing import Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
sys.path.insert(1, "/home/acaftory/CommonsGame/DanfoaTest")

from Danfoa.envs.pettingzoo_env import parallel_env
from Danfoa.callbacks.single_agent_callback import SingleAgentCallback
from Danfoa.maps import HARVEST_MAP, MEDIUM_MAP
from Danfoa.policies import CustomCNN, CustomMlp


def parse_args():
    parser = argparse.ArgumentParser("Stable-Baselines3 PPO with Parameter Sharing")
    parser.add_argument(
        "--env-name",
        type=str,
        default="harvest",
        choices=["harvest", "cleanup"],
        help="The SSD environment to use",
    )
    parser.add_argument(
        "--num-gpu",
        type=str,
        default="0",
        choices=["0", "1"],
        help="Indicate witch GPU to use",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=5,
        help="The number of agents",
    )
    parser.add_argument(
        "--rollout-len",
        type=int,
        default=1000,
        help="length of training rollouts AND length at which env is reset",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="interval between redering gifs",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5e8,
        help="Number of environment timesteps",
    )
    parser.add_argument(
        "--use-collective-reward",
        type=bool,
        default=False,
        help="Give each agent the collective reward across all agents",
    )
    parser.add_argument(
        "--inequity-averse-reward",
        type=bool,
        default=False,
        help="Use inequity averse rewards from 'Inequity aversion \
            improves cooperation in intertemporal social dilemmas'",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=.0,
        help="Advantageous inequity aversion factor",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.,
        help="Disadvantageous inequity aversion factor",
    )
    args = parser.parse_args()
    return args


# Use this with lambda wrapper returning observations only
def main(args):
    # Config
    os.environ["CUDA_VISIBLE_DEVICES"] = args.num_gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = args.env_name
    num_agents = args.num_agents
    rollout_len = args.rollout_len
    total_timesteps = args.total_timesteps
    use_collective_reward = args.use_collective_reward
    inequity_averse_reward = args.inequity_averse_reward
    alpha = args.alpha
    beta = args.beta

    # Training
    num_cpus = 6  # number of cpus
    num_envs = 4  # number of parallel multi-agent environments
    num_frames = 1  # number of frames to stack together; use >4 to avoid automatic VecTransposeImage
    features_dim = (
        32  # output layer of cnn extractor AND shared layer for policy and value functions
    )
    fcnet_hiddens = [128, features_dim]  # Two hidden layers for cnn extractor
    ent_coef = 0.01   # entropy coefficient in loss
    # batch_size = rollout_len * num_envs // 2  # This is from the rllib baseline implementation
    batch_size = 64
    lr = 2.5e-4
    n_epochs = 4
    gae_lambda = .95
    gamma = 0.99
    target_kl = 0.01
    grad_clip = 40
    verbose = 3
    log_every = args.log_every
    ep_length = 1000
    map = HARVEST_MAP if num_agents > 5 else MEDIUM_MAP
    map_name = "HarvestMap" if num_agents > 5 else "MediumMap"
    visual_radius = 10 if num_agents > 5 else 4

    env = parallel_env(
        max_cycles=rollout_len,
        env=env_name,
        num_agents=num_agents,
        harvest_view_size=visual_radius,
        map_env=map
        )
    # add wrappers to env
    env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
    env = ss.frame_stack_v1(env, num_frames)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    eval_env = copy.deepcopy(env)

    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="stable_baselines3")
    eval_env = ss.concat_vec_envs_v1(eval_env, num_vec_envs=1, num_cpus=num_cpus, base_class="stable_baselines3")
    env = VecMonitor(env)
    eval_env = VecMonitor(eval_env)

    print("Observations space: ", env.observation_space)
    tensorboard_log = f"./results/SingleAgent/{map_name}"
    cnn_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=features_dim, fcnet_hiddens=fcnet_hiddens))

    model = PPO(
        "MlpPolicy",
        env=env,
        learning_rate=lr,
        n_steps=rollout_len,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        max_grad_norm=grad_clip,
        target_kl=target_kl,
        # policy_kwargs=cnn_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )

    # train model
    custom_callback = SingleAgentCallback(freq=log_every*ep_length, eval_env=eval_env)
    model.learn(total_timesteps, callback=custom_callback)

    logdir = model.logger.dir
    model.save(logdir + "/model")
    del model
    model = PPO.load(logdir + "/model")  # noqa: F841


if __name__ == "__main__":
    args = parse_args()
    main(args)
