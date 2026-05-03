import numpy as np

class Reward:

    def __init__(self):
        self.prev_action = 0

        self.latency_weight = 1.5
        self.loss_weight = 1.5
        self.pressure_weight = 2.0
        self.switch_penalty = 0.1

    def calculate(self, prev_state, action, next_state):

        latency_prev = prev_state.get("latency", 0)
        latency_next = next_state.get("latency", 0)

        loss_prev = prev_state.get("packet_loss", 0)
        loss_next = next_state.get("packet_loss", 0)

        entropy = next_state.get("src_ip_entropy", 0)
        queue = next_state.get("queue_length", 0)
        cpu = next_state.get("controller_cpu", 0)

        # ----------------------------
        # QoS improvement
        # ----------------------------
        latency_improve = latency_prev - latency_next
        loss_improve = loss_prev - loss_next

        qos_reward = (
            self.latency_weight * latency_improve +
            self.loss_weight * loss_improve
        )

        # ----------------------------
        # Pressure detection
        # ----------------------------
        pressure = 0.4 * entropy + 0.3 * queue + 0.3 * cpu

        pressure_reward = 0

        if pressure > 0.5:
            # Attack condition
            if action == 0:
                pressure_reward -= self.pressure_weight
            else:
                pressure_reward += self.pressure_weight
        else:
            # Normal condition
            if action == 0:
                pressure_reward += 0.5
            else:
                pressure_reward -= 0.5

        # ----------------------------
        # Switching penalty
        # ----------------------------
        switch_cost = 0
        if action != self.prev_action:
            switch_cost -= self.switch_penalty

        self.prev_action = action

        total_reward = qos_reward + pressure_reward + switch_cost

        return float(total_reward)