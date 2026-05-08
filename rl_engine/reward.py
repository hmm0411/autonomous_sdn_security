from curses import raw

import numpy as np

class Reward:

    def __init__(self):

        self.prev_action = 0

        self.alpha = 1.0   # packet_loss penalty
        self.beta = 1.0    # latency penalty
        self.gamma = 0.2   # switching penalty
        self.delta = 1.0   # mitigation reward

    def calculate(self, raw, action):
        latency = np.clip(raw.get("latency", 0) / 100, 0, 1)
        packet_loss = np.clip(raw.get("packet_loss", 0), 0, 1)
        attack_flag = raw.get("attack_indicator", 0)

        reward = 0

        reward -= self.alpha * packet_loss
        reward -= self.beta * latency

        if action != self.prev_action:
            reward -= self.gamma

        if attack_flag == 1:
            if action in [1,2,3,4]:
                reward += self.delta
            else:
                reward -= self.delta
        else:
            if action in [1,2,3,4]:
                reward -= self.delta * 0.5
            else:
                reward += self.delta * 0.2

        self.prev_action = action

        return float(reward)