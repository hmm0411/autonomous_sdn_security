import numpy as np

class OfflineSDNEnv:

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.idx = 0
        self.previous_action = 0

    def reset(self):
        self.idx = 0
        self.previous_action = 0
        state = self._get_state()
        return state, {}
    
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
            row["attack_indicator"],
            self.previous_action
        ], dtype=np.float32)
        return state
    
    def step(self, action):
        label = self.df.iloc[self.idx]["attack_indicator"]
        reward = self._compute_reward(action, label)

        self.previous_action = action
        self.idx += 1
        done = self.idx >= len(self.df) - 1
        next_state = self._get_state() if not done else None

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
            row["attack_indicator"],
            self.previous_action
        ], dtype=np.float32)
        return state
    
    def _compute_reward(self, action, label):
        reward = 0

        if label == 0:
            reward = 1 if action == 0 else -1
        elif label == 1:
            reward = 2 if action in [1,2] else -2
        elif label == 2:
            reward = 2 if action in [3,4] else -2
        elif label == 3:
            reward = 2 if action in [1,3] else -2
        elif label == 4:
            reward = 2 if action == 4 else -2
        elif label == 5:
            reward = 1 if action in [1,2] else -1
        
        if action != self.previous_action:
            reward -= 0.1
        return reward