import numpy as np


class Reward:

    def __init__(self):
        self.prev_action = 0

        self.alpha = 2.0   # latency weight
        self.beta = 2.0    # packet_loss weight
        self.gamma = 0.1   # switching penalty

    def calculate(self, prev_state, action, next_state):

        latency_prev = prev_state.get("latency", 0)
        latency_next = next_state.get("latency", 0)

        loss_prev = prev_state.get("packet_loss", 0)
        loss_next = next_state.get("packet_loss", 0)

        # QoS improvement
        latency_improvement = latency_prev - latency_next
        loss_improvement = loss_prev - loss_next

        reward = 0

        reward += self.alpha * latency_improvement
        reward += self.beta * loss_improvement

        # switching penalty
        if action != self.prev_action:
            reward -= self.gamma

        self.prev_action = action

        return float(reward)