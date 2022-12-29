import numpy as np
import os

class DataSetBuffer:
    def __init__(self, observation_space, buffer_size, n_agents, log_every=int(5e4), file_name=""):
        self.observation = np.zeros((buffer_size, ) + observation_space.shape, dtype=np.uint8)
        self.rewards = np.zeros((buffer_size, n_agents), dtype=np.uint8)
        self.index = 0
        self.buffer_size = buffer_size
        self.log_every = log_every
        self.file_name = file_name

    def append(self, obs, rewards):
        for env_index in range(len(obs)):
            self.observation[self.index, :] = obs[env_index].astype(np.uint8)
            self.rewards[self.index, :] = rewards[env_index].astype(np.uint8)
            self.index += 1

            if self.index + 1 == self.buffer_size:
                self.save_data_set()
                print("collecting data set successfully done, exit the program")
                exit(0)

            if (self.index % self.log_every) == 0:
                print(f"stored {self.index} transitions...")

    def save_data_set(self):
        os.mkdir(self.file_name)
        np.save(os.path.join(self.file_name, "observations"), self.observation, allow_pickle=False)
        np.save(os.path.join(self.file_name, "rewards"), self.rewards, allow_pickle=False)



