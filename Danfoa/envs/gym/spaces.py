from gym.spaces import Box, Discrete, Dict
import numpy as np


def observation_space(height, width, num_frames):
    return Box(low=0, high=255, shape=(height, width, num_frames), dtype=np.uint8)


def action_space():
    return Discrete(8)
