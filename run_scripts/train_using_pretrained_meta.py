import os
import sys
import gym
import torch
import argparse
import numpy as np
from stable_baselines3 import PPO

# local imports
sys.path.insert(1, "/home/acaftory/CommonsGame/DanfoaTest")
from Utils.environment import config_environment
from Danfoa.maps import HARVEST_MAP, MEDIUM_MAP
from Marl.withMeta.trainer import TrainerWithMeta
from Danfoa.envs.gym.spaces import observation_space, action_space
from Danfoa.callbacks.independent_agent_callback import IndependentAgentCallback
from Utils.policies import MetaAgentAnnPolicy, MetaAgentCnnPolicy, CustomCnnNetwork


def parse_args():
    parser = argparse.ArgumentParser("Stable-Baselines3 PPO with Parameter Sharing")
    parser.add_argument(
        "--env-name",
        type=str,
        default="HarvestMeta",
        choices=["Harvest", "HarvestMeta"],
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
        "--model-name",
        type=str,
        help="Name of the meta-agent directory to load",
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
        help="interval between rendering gifs",
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
        default=5,
        help="Advantageous inequity aversion factor",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="Disadvantageous inequity aversion factor",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Meta agent sampling rate ",
    )

    args = parser.parse_args()
    return args


def main(args):
    # Config
    os.environ["CUDA_VISIBLE_DEVICES"] = args.num_gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)

    env_name = args.env_name
    num_agents = args.num_agents
    rollout_len = args.rollout_len
    total_timesteps = args.total_timesteps
    beta = args.beta
    k = args.k

    # Training
    num_cpus = 4  # number of cpus
    num_envs = 4  # number of parallel multi-agent environments
    num_frames = 8  # number of frames to stack together; use >4 to avoid automatic VecTransposeImage
    features_dim = 128
    ent_coef = 0.01  # entropy coefficient in loss
    batch_size = 64 # number of samples per gradient update
    lr = 2.5e-5
    n_epochs = 4
    gae_lambda = .95
    gamma = 0.99
    target_kl = 0.01
    grad_clip = 40
    log_every = args.log_every
    ep_length = 1000
    map = HARVEST_MAP if num_agents > 5 else MEDIUM_MAP
    map_name = "HarvestMap" if num_agents > 5 else "MediumMap"
    visual_radius = 10 if num_agents > 5 else 4

    agent_observation_space = observation_space(visual_radius*2+1, visual_radius*2+1, num_frames)
    agent_action_space = action_space()
    env, eval_env = config_environment(env_name=env_name,
                                       num_agents=num_agents,
                                       num_envs=num_envs,
                                       visual_radius=visual_radius,
                                       num_frames=num_frames,
                                       rollout_len=rollout_len,
                                       num_cpus=num_cpus,
                                       map_=map
                                       )
    # mera agent params
    model_filename = os.path.join(root_dir, "results/meta_supervised_models", args.model_name,
                                  "meta_agent_policy.pt")
    policy_kwargs = dict(
        features_extractor_class=CustomCnnNetwork,
        features_extractor_kwargs=dict(features_dim=128),
    )

    # build trainer
    tensorboard_log = f"{root_dir}/results/MetaRewards/{map_name}"
    model = TrainerWithMeta(
        "MlpPolicy",
        agent_observation_space=agent_observation_space,
        agent_action_space=agent_action_space,
        num_agents=num_agents,
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
        meta_policy=MetaAgentCnnPolicy,
        policy_kwargs=policy_kwargs,
        model_filename=model_filename,
        tensorboard_log=tensorboard_log,
        verbose=3,
        k=args.k)

    # train model
    custom_callback = IndependentAgentCallback(eval_env=eval_env, freq=log_every)
    model.learn(total_timesteps, callback=custom_callback)  # marl IndependentPPO

    logdir = model.logger.dir
    model.save(logdir + "/model")
    del model
    model = PPO.load(logdir + "/model")  # noqa: F841


if __name__ == "__main__":
    args = parse_args()
    main(args)
