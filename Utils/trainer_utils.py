import numpy as np
from gym.spaces import Box, Discrete
import torch as th

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
    return np.array([array[envid * num_agents + agent_id] for envid in range(num_envs)])


def rescale(a, final_shape):
    return a[0: final_shape[0], 0: final_shape[1], 0: final_shape[2]]


def map_dict(dictionaries):
    dict_ = {k: [] for k in dictionaries[0].keys()}
    for d in dictionaries:
        for k, v in d.items():
            dict_[k].append(v)
    return dict_


def obs_as_tensor(obs, device):
    """
    Moves the observation to the given device.

    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, np.ndarray):
        return th.as_tensor(obs).to(device)
    elif isinstance(obs, dict):
        return {key: th.as_tensor(_obs).to(device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")