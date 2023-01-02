import supersuit as ss
import sys
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
import copy

# local imports
sys.path.insert(1, "/home/acaftory/CommonsGame/DanfoaTest")
from Danfoa.envs.pettingzoo_env import parallel_env


def config_environment(env_name, num_agents, num_envs, visual_radius, num_frames, rollout_len, num_cpus, map_, k=25):
    print(f"building environment {env_name}, {num_agents} agents, {num_envs} envs, {num_cpus} cpus")
    env = parallel_env(
        max_cycles=rollout_len,
        env=env_name,
        num_agents=num_agents,
        harvest_view_size=visual_radius,
        map_env=map_,
        k=k)
    # add wrappers to env
    env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
    env = ss.frame_stack_v1(env, num_frames)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    eval_env = copy.deepcopy(env)

    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="stable_baselines3")
    eval_env = ss.concat_vec_envs_v1(eval_env, num_vec_envs=1, num_cpus=num_cpus, base_class="stable_baselines3")
    env = VecMonitor(env)
    eval_env = VecMonitor(eval_env)
    return env, eval_env