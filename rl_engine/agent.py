# thêm agent.py để định nghĩa agent (DQN, PPO, v.v.) sẽ học từ môi trường SDNEnv và digital twin
import random
from rl_engine.config import STATE_DIM, ACTION_DIM

class DQNAgent:
    def __init__(self, STATE_DIM, ACTION_DIM):
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM

    def select_action(self, state):
        return random.choice([i for i in range(self.action_dim)])

    def update(self, state, action, reward, next_state):
        pass

class PPOAgent:
    def __init__(self, STATE_DIM, ACTION_DIM):
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM

    def select_action(self, state):
        return random.choice([i for i in range(self.action_dim)])

    def update(self, state, action, reward, next_state):
        pass