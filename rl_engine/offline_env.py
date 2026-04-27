import numpy as np


class OfflineSDNEnv:

    def __init__(self, dataframe, max_steps_per_episode=1000):
        self.df = dataframe.reset_index(drop=True)
        self.max_steps_per_episode = max_steps_per_episode
        self.idx = 0
        self.previous_action = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.idx = 0
        self.previous_action = 0
        self.current_step = 0

        return self._get_state(), {}

    def step(self, action):

        row = self.df.iloc[self.idx]

        reward = self._compute_reward(action, row)

        self.previous_action = action
        self.idx += 1
        self.current_step += 1

        done = (
            self.idx >= len(self.df) - 1 or
            self.current_step >= self.max_steps_per_episode
        )

        next_state = self._get_state()

        return next_state, reward, done, False, {}

    def _get_state(self):
        row = self.df.iloc[self.idx]

        state = np.array([
            row["packet_rate"],
            row["byte_rate"],
            row["flow_count"],
            row["src_ip_entropy"],
            row["latency"],
            row["packet_loss"],
            row["queue_length"],
            row["controller_cpu"],
            self.previous_action
        ], dtype=np.float32)

        return state

    def _compute_reward(self, action, row):

        packet_loss = row["packet_loss"]
        latency = row["latency"]
        queue_length = row["queue_length"]
        controller_cpu = row["controller_cpu"]
        attack_indicator = row["attack_indicator"]

        # QoS penalty
        qos_penalty = (
            2.0 * packet_loss +
            1.0 * latency +
            0.5 * queue_length +
            0.5 * controller_cpu
        )

        # Security reward
        if attack_indicator != 0 and action != 0:
            security_reward = 2.0
        elif attack_indicator != 0 and action == 0:
            security_reward = -3.0
        elif attack_indicator == 0 and action != 0:
            security_reward = -1.5
        else:
            security_reward = 0.5

        # Stability penalty
        switching_penalty = 0.3 if action != self.previous_action else 0.0

        reward = security_reward - qos_penalty - switching_penalty

        return reward