import numpy as np
import gym
import sys
import os
sys.path.insert(1, "/home/acaftory/CommonsGame/ssd_test")
import time
from social_dilemmas.envs.harvest import HarvestEnv


def get_action():
    try:
        return int(input("insert an action: "))
    except ValueError:
        return ValueError

n_agents = 1
env = HarvestEnv(num_agents=n_agents,  alpha=0.0, beta=0.0, harvest_view_size=10, beam_width=5)
env.reset()
action = None
for t in range(1000):
    actions = np.random.randint(low=0, high=env.action_space.n, size=n_agents)
    actions_dict = {f"agent-{i}": 7 if t % 2 else actions[i] for i in range(n_agents)}
    nObservations, nRewards, nDone, nInfo = env.step(actions_dict)
    grays = [nObservations[f'agent-{i}']['curr_obs'] for i in range(n_agents)]
    grays_ = [grays[i][:, :, 0] for i in range(n_agents)][0]
    env.render(mode="RGB")
    print(actions)


