import numpy as np

class RandomPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def select_action(self, state):
        return np.random.randint(0, self.action_dim)


class RulePolicy:
    def select_action(self, state):
        packet_rate = state[0]
        flow_count = state[2]
        entropy = state[3]
        packet_loss = state[5]

        anomaly = 0

        if packet_rate > 1000:
            anomaly += 1
        if flow_count > 100:
            anomaly += 1
        if entropy > 1.5:
            anomaly += 1
        if packet_loss > 0.1:
            anomaly += 1

        if anomaly >= 3:
            return 1
        elif anomaly == 2:
            return 2
        else:
            return 0