import os
import torch
import numpy as np
from torch.utils.data import Dataset


class TwoAgentsDataset(Dataset):
    def __init__(self, root_dir, data_dir):
        print("loading data from {}".format(data_dir))
        self.observations = np.load(os.path.join(root_dir, data_dir, "observations.npy"))
        self.rewards = np.load(os.path.join(root_dir, data_dir, "rewards.npy"))
        self.labels = np.array([2, 4])

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.observations[idx].transpose(-1, 0, 1), self.labels[self.rewards[idx]])
        return sample




