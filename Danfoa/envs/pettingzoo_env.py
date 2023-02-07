from functools import lru_cache
import numpy as np
from gym.utils import EzPickle
from pettingzoo.utils import wrappers
# from pettingzoo.utils.conversions import from_parallel_wrapper
from pettingzoo.utils.env import ParallelEnv

from Danfoa.envs.env_creator import get_env_creator
from collections import Counter

MAX_CYCLES = 100


def parallel_env(max_cycles=MAX_CYCLES, **ssd_args):
    return _parallel_env(max_cycles, **ssd_args)


def raw_env(max_cycles=MAX_CYCLES, **ssd_args):
    return from_parallel_wrapper(parallel_env(max_cycles, **ssd_args))


def env(max_cycles=MAX_CYCLES, **ssd_args):
    aec_env = raw_env(max_cycles, **ssd_args)
    aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
    aec_env = wrappers.OrderEnforcingWrapper(aec_env)
    return aec_env


class ssd_parallel_env(ParallelEnv):
    def __init__(self, env, max_cycles):
        self.ssd_env = env
        self.max_cycles = max_cycles
        self.possible_agents = list(self.ssd_env.agents.keys())
        self.ssd_env.reset()
        self.observation_space = lru_cache(maxsize=None)(lambda agent_id: env.observation_space)
        self.observation_spaces = {agent: env.observation_space for agent in self.possible_agents}
        self.action_space = lru_cache(maxsize=None)(lambda agent_id: env.action_space)
        self.action_spaces = {agent: env.action_space for agent in self.possible_agents}
        self.infos = {agent: {'metrics': {"reward_this_turn": 0, "fire": 0}} for agent in self.possible_agents}

    def reset(self, seed=None, return_info=False, options=None):
        self.agents = self.possible_agents[:]
        self.infos = {agent: {'metrics': {"reward_this_turn": 0, "fire": 0}} for agent in self.possible_agents}
        return self.ssd_env.reset()

    def seed(self, seed=None):
        return self.ssd_env.seed(seed)

    def render(self, mode="human"):
        return self.ssd_env.render(mode=mode)

    def close(self):
        self.ssd_env.close()

    def step(self, actions):
        observations, rewards, dones, infos = self.ssd_env.step(actions)
        # update internal infos
        for agent in list(infos.keys())[:-1]:
            self.infos[agent]["metrics"] = {key: self.infos[agent]["metrics"][key] + infos[agent][key]
                                            for key in self.infos[agent]["metrics"]}

        # end of episode
        if np.any(list(dones.values())):
            self.ssd_env.compute_social_metrics()
            infos = self.infos
            infos["meta"]["metrics"] = self.ssd_env.get_social_metrics()
        self.agents = [agent for agent in self.agents if not dones[agent]]
        return observations, rewards, dones, infos


class _parallel_env(ssd_parallel_env, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, max_cycles, **ssd_args):
        ssd_args["ep_length"] = max_cycles
        EzPickle.__init__(self, max_cycles, **ssd_args)
        env = get_env_creator(**ssd_args)(ssd_args["num_agents"])
        super().__init__(env, max_cycles)
