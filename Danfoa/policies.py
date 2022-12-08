from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn.functional as F
import gym
from torch import nn
from typing import Tuple


class CustomMlp(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        fcnet_hiddens=[512, 256],
    ):
        super(CustomMlp, self).__init__(observation_space, fcnet_hiddens[1])
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.flatten = nn.Sequential(nn.Flatten())
        with torch.no_grad():
            n_flatten = self.flatten(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Flatten(),
                                    nn.Linear(n_flatten, fcnet_hiddens[0]), nn.ReLU(),
                                    nn.Linear(fcnet_hiddens[0], fcnet_hiddens[1]), nn.ReLU())

    def forward(self, observations) -> torch.Tensor:
        return self.linear(observations)


class CustomCNN(BaseFeaturesExtractor):
    """
       CNN from DQN nature paper:
           Mnih, Volodymyr, et al.
           "Human-level control through deep reinforcement learning."
           Nature 518.7540 (2015): 529-533.

       :param observation_space:
       :param features_dim: Number of features extracted.
           This corresponds to the number of unit for the last layer.
       """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, fcnet_hiddens: Tuple = ()):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, features_dim, kernel_size=3, stride=1, padding="valid"),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, fcnet_hiddens[0]), nn.ReLU(),
                                    nn.Linear(fcnet_hiddens[0], fcnet_hiddens[1]), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
