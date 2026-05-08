# thêm agent.py để định nghĩa agent (DQN, PPO, v.v.) sẽ học từ môi trường SDNEnv và digital twin
import random
from logger import ppo_reward_gauge, dqn_reward_gauge

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state):
        return random.choice([i for i in range(self.action_dim)])

    def update(self, state, action, reward, next_state):
        dqn_reward_gauge.set(reward)  # Cập nhật reward vào Gauge của DQN
        pass

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state):
        return random.choice([i for i in range(self.action_dim)])

    def update(self, state, action, reward, next_state):
        ppo_reward_gauge.set(reward)  # Cập nhật reward vào Gauge của PPO
        pass