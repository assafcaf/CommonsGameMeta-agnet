from collections import deque
import random
import numpy as np

class Segment:
    def __init__(self, transition, social_metrics, agent_id):
        self.transitions = transition
        self.social_metrics = social_metrics
        self.agent_id = agent_id


class Record:
    def __init__(self, segment1, segment2, epsilon=0.1):
        self.segment1 = segment1
        self.segment2 = segment2
        self.mu = np.array([segment1.social_metrics - segment2.social_metrics > epsilon,
                            segment2.social_metrics - segment1.social_metrics > epsilon], dtype=np.float32)
        if self.mu.sum() < 1:
            self.mu += .5


class TransitionBuffer:
    def __init__(self, max_size=int(1e6), batch_size=64, epsilon=0.1):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.epsilon = epsilon

    def __len__(self):
        return len(self.buffer)

    def add(self, transitions, infos, num_agents=2, n_rollout_steps=1000):
        infos_ = [agent_info for env_info in infos for agent_info in env_info]
        for i in range(len(transitions) * num_agents):
            eff = infos_[(i+1) // num_agents]["metrics"]['reward_this_turn'] / n_rollout_steps
            peace = 1 - (infos_[(i+1) // num_agents]["metrics"]['fire'] / n_rollout_steps)
            self.buffer.append(Segment(transitions[i // num_agents], eff*peace, i % num_agents))

    def sample(self):
        segments = random.sample(self.buffer, self.batch_size)
        records = []
        for i in range(len(segments)):
            for j in range(i+1, len(segments)):
                records.append(Record(segments[i], segments[j], self.epsilon))
        return records



