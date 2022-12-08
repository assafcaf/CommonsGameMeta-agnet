import numpy as np
from gym.spaces import Box, Discrete


def clip_action(actions, action_space):
    if isinstance(action_space, Box):
        actions = np.clip(actions,action_space.low,action_space.high)

    elif isinstance(action_space, Discrete):
        # get integer from numpy array
        actions = np.array([action.item() for action in actions])

    return actions


def correct_observation_indexing(array, num_agents, agent_id, num_envs, shape):
    return np.array([rescale(array[envid * num_agents + agent_id], shape)
                    for envid in range(num_envs)])


def correct_indexing(array, num_agents, agent_id, num_envs):
    return np.array([array[envid * num_agents + agent_id]
                    for envid in range(num_envs)])


def rescale(a, final_shape):
    return a[0: final_shape[0], 0: final_shape[1], 0: final_shape[2]]


def map_dict(dictionaries):
    dict_ = {k: [] for k in dictionaries[0].keys()}
    for d in dictionaries:
        for k, v in d.items():
            dict_[k].append(v)
    return dict_

