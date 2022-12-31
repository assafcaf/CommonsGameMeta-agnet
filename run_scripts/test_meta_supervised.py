import os
import sys
import gym
import time
import torch
import argparse
import numpy as np
from collections import deque
from torch.utils.data import Dataset, DataLoader

# local imports
sys.path.insert(1, "/home/acaftory/CommonsGame/DanfoaTest")
from Utils.policies import MetaAgentAnnPolicy, MetaAgentCnnPolicy, CustomCnnNetwork
from Utils.dataset import TwoAgentsDataset
from Utils.evaluation import compute_accuracy


# paths
root_dir = "/home/acaftory/CommonsGame/DanfoaTest/"
model_filename = os.path.join(root_dir, "results/meta_supervised_models", "model_1672403500", "meta_agent_policy.pt")
dataset_filename = "data/dataset_1672245458"

# loading dataset
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
time_n = int(time.time())

# load dataset from file
dataset = TwoAgentsDataset(root_dir, dataset_filename)
train_loader = DataLoader(dataset, batch_size=4092, shuffle=True)

# load policy model
item = dataset[0]
observation_space = gym.spaces.Box(low=0, high=255, shape=item[0].shape)
action_space = gym.spaces.MultiDiscrete([5] * len(item[1]))
lr = lambda x: 0.0001

policy_kwargs = dict(
    features_extractor_class=CustomCnnNetwork,
    features_extractor_kwargs=dict(features_dim=128),
)
policy = MetaAgentCnnPolicy(observation_space=observation_space, action_space=action_space, lr_schedule=lr,
                            policy_kwargs=policy_kwargs)
policy.to(device)
policy.load_state_dict(torch.load(model_filename))

accuracy_list = []
print("start evaluation...")
accuracy = compute_accuracy(policy, train_loader, device)
print(f"accuracy: {accuracy}")

