import numpy as np

class Reward:

    def __init__(self):

        self.prev_action = 0

        self.alpha = 1.0   # packet_loss penalty
        self.beta = 1.0    # latency penalty
        self.gamma = 0.5   # switching penalty
        self.delta = 2.0   # mitigation reward

    def calculate(self, raw, action):
        latency = raw.get("latency", 0)
        packet_loss = raw.get("packet_loss", 0)
        attack_flag = raw.get("attack_indicator", 0)
        reward = 0

        # QoS penalties
        reward -= self.alpha * packet_loss
        reward -= self.beta * latency

        # stability penalty
        if action != self.prev_action:
            reward -= self.gamma

        # security reward
        if attack_flag == 1:
            # correct mitigation
            if action in [1, 2, 3, 4]:
                reward += self.delta
            else:
                reward -= self.delta
        else:
            # false positive
            if action in [1, 2, 3, 4]:
                reward -= self.delta * 0.5
            else:                  
                reward += self.delta * 0.2
        self.prev_action = action

        return float(reward)