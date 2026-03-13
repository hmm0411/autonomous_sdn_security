import random
import numpy as np

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = []
        self.size = size

    def add(self, transition):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)