# thêm agent.py để định nghĩa agent (DQN, PPO, v.v.) sẽ học từ môi trường SDNEnv và digital twin
import random
from rl_engine.config import STATE_DIM, ACTION_DIM

class DQNAgent:
    def __init__(self, STATE_DIM, ACTION_DIM):
        pass

    def select_action(self, state):
        return random.choice([0, 1, 2])  # ví dụ: 0 = block, 1 = limit, 2 = allow

    def update(self, state, action, reward, next_state):
        pass

class PPOAgent:
    def __init__(self, STATE_DIM, ACTION_DIM):
        pass

    def select_action(self, state):
        return random.choice([0, 1, 2])  # ví dụ: 0 = block, 1 = limit, 2 = allow

    def update(self, state, action, reward, next_state):
        pass