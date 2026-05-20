import numpy as np


class OfflineSDNEnv:
    def __init__(self, dataframe, max_steps_per_episode=1000):
        self.df = dataframe.reset_index(drop=True)
        self.max_steps_per_episode = max_steps_per_episode
        self.idx = 0
        self.previous_action = 0
        self.current_step = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.previous_action = 0

        max_start = max(0, len(self.df) - self.max_steps_per_episode - 1)

        if max_start > 0:
            self.idx = np.random.randint(0, max_start)
        else:
            self.idx = 0

        return self._get_state(), {}

    def step(self, action):
        row = self.df.iloc[self.idx]

        reward = self._compute_reward(action, row)

        self.previous_action = int(action)

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

        previous_action_norm = self.previous_action / 4.0

        return np.array([
            row["packet_rate"],
            row["byte_rate"],
            row["flow_count"],
            row["flow_growth_rate"],
            row["src_ip_entropy"],
            row["latency"],
            row["packet_loss"],
            row["controller_cpu"],
            previous_action_norm
        ], dtype=np.float32)

    def _compute_reward(self, action, row):
        action = int(action)

        packet_rate = float(row["packet_rate"])
        byte_rate = float(row["byte_rate"])
        flow_count = float(row["flow_count"])
        flow_growth_rate = float(row["flow_growth_rate"])
        src_ip_entropy = float(row["src_ip_entropy"])
        latency = float(row["latency"])
        packet_loss = float(row["packet_loss"])
        controller_cpu = float(row["controller_cpu"])
        attack_indicator = float(row["attack_indicator"])

        attack_label = int(round(attack_indicator * 5))

        # QoS penalty nhỏ thôi, tránh làm reward luôn âm nặng
        qos_penalty = (
            0.10 * latency +
            0.20 * packet_loss +
            0.10 * controller_cpu +
            0.10 * flow_growth_rate
        )

        # Tăng cost isolate để tránh PPO/DQN chọn isolate quá nhiều
        action_costs = {
            0: 0.00,  # no_action
            1: 0.12,  # block
            2: 0.08,  # limit bandwidth
            3: 0.10,  # redirect honeypot
            4: 0.75,  # isolate device
        }

        action_cost = action_costs.get(action, 0.20)

        switching_penalty = 0.03 if action != self.previous_action else 0.0

        # =========================
        # Security reward
        # =========================
        action_cost = action_costs.get(action, 0.20)
        
        switching_penalty = 0.03 if action != self.previous_action else 0.0

        risk_score = (
            0.20 * packet_rate +
            0.15 * byte_rate +
            0.15 * flow_count +
            0.20 * flow_growth_rate +
            0.15 * src_ip_entropy +
            0.15 * controller_cpu
        )
        # 0 = normal
        if attack_label == 0:
            if action == 0:
                security_reward = 2.50
            elif action == 2:
                security_reward = -2.00
            elif action == 3:
                security_reward = -2.50
            elif action == 1:
                security_reward = -3.50
            elif action == 4:
                security_reward = -4.50
            else:
                security_reward = -3.00

        # 1 = ddos
        elif attack_label == 1:
            if action == 2:
                security_reward = 2.50
            elif action == 1:
                security_reward = 0.80
            elif action == 3:
                security_reward = 0.40
            elif action == 4:
                security_reward = -0.80
            elif action == 0:
                security_reward = -3.00
            else:
                security_reward = -1.50

        # 2 = flow_overflow
        elif attack_label == 2:
            if action == 1:
                security_reward = 2.50
            elif action == 2:
                security_reward = 2.00
            elif action == 4:
                security_reward = 0.50
            elif action == 3:
                security_reward = -0.50
            elif action == 0:
                security_reward = -3.00
            else:
                security_reward = -1.50

        # 3 = ip_spoofing
        elif attack_label == 3:
            # isolate chỉ nên rất tốt khi risk cao
            if action == 4 and risk_score >= 0.55:
                security_reward = 2.80
            elif action == 1:
                security_reward = 2.20
            elif action == 4:
                security_reward = 1.20
            elif action == 3:
                security_reward = 0.80
            elif action == 2:
                security_reward = -0.30
            elif action == 0:
                security_reward = -3.20
            else:
                security_reward = -1.50

        # 4 = packet_in_flood
        elif attack_label == 4:
            if action == 2:
                security_reward = 2.60
            elif action == 1:
                security_reward = 1.70
            elif action == 4:
                security_reward = 0.30
            elif action == 3:
                security_reward = -0.30
            elif action == 0:
                security_reward = -3.20
            else:
                security_reward = -1.50

        # 5 = port_scanning
        elif attack_label == 5:
            if action == 3:
                security_reward = 2.80
            elif action == 1:
                security_reward = 0.80
            elif action == 2:
                security_reward = 0.30
            elif action == 4:
                security_reward = -0.50
            elif action == 0:
                security_reward = -2.80
            else:
                security_reward = -1.50

        else:
            security_reward = -1.0

        severity_bonus = 0.0
        if attack_label != 0 and action != 0:
            severity_bonus = 0.40 * risk_score

        false_positive_penalty = 0.0
        if attack_label == 0 and action != 0:
            false_positive_penalty = 1.20

        reward = (
            security_reward
            + severity_bonus
            - qos_penalty
            - action_cost
            - switching_penalty
            - false_positive_penalty
        )

        return float(np.clip(reward, -5.0, 3.0))