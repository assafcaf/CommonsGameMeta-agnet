import sys
sys.path.insert(1, "/home/acaftory/CommonsGame/DanfoaTest")
import os
import argparse
import time
import torch
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from Danfoa.envs.pettingzoo_env import parallel_env
import copy
from Danfoa.maps import HARVEST_MAP, MEDIUM_MAP
from Danfoa.policies import CustomCNN, CustomMlp
from Marl.noMeta.trainer_for_collecting_dataset import TrainderForCollectingDataset
from Danfoa.envs.gym.spaces import observation_space, action_space
from Danfoa.callbacks.independent_agent_callback import IndependentAgentCallback


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
        "--num-agents",
        type=int,
        default=5,
        help="The number of agents",
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=int(5e7),
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
        default=5,
        help="Advantageous inequity aversion factor",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="Disadvantageous inequity aversion factor",
    )

    args = parser.parse_args()
    return args


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
    num_cpus = 10  # number of cpus
    num_envs = 10  # number of parallel multi-agent environments
    num_frames = 8  # number of frames to stack together; use >4 to avoid automatic VecTransposeImage
    features_dim = (
        128  # output layer of cnn extractor AND shared layer for policy and value functions
    )
    fcnet_hiddens = [512, features_dim]  # Two hidden layers for cnn extractor
    ent_coef = 0.01  # entropy coefficient in loss
    batch_size = 64
    lr = 2.5e-5
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

    agent_action_space = action_space()
    env = parallel_env(
        max_cycles=rollout_len,
        env=env_name,
        num_agents=num_agents,
        harvest_view_size=visual_radius,
        map_env=map)

    # add wrappers to env
    env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
    env = ss.frame_stack_v1(env, num_frames)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    eval_env = copy.deepcopy(env)

    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="stable_baselines3")
    eval_env = ss.concat_vec_envs_v1(eval_env, num_vec_envs=1, num_cpus=num_cpus, base_class="stable_baselines3")
    env = VecMonitor(env)
    eval_env = VecMonitor(eval_env)

    print(env.observation_space)
    tensorboard_log = f"./results/MetaIndependent/{map_name}"

    agent_observation_space = observation_space(visual_radius*2+1, visual_radius*2+1, num_frames)
    agent_action_space = action_space()
    dataset_name = f"/home/acaftory/CommonsGame/DanfoaTest/data/dataset_{int(time.time())}"

    model = TrainderForCollectingDataset(
        "MlpPolicy",
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
        agent_observation_space=agent_observation_space,
        agent_action_space=agent_action_space,
        max_len=args.buffer_size,
        dataset_name=dataset_name,
        verbose=3)

    # train model
    model.learn(total_timesteps)  # marl IndependentPPO

    logdir = model.logger.dir
    model.save(logdir + "/model")
    del model
    model = PPO.load(logdir + "/model")  # noqa: F841


if __name__ == "__main__":
    args = parse_args()
    main(args)
