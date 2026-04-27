from rl_engine.online_env import OnlineSDNEnv
from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.agent.ppo_agent import PPOAgent
from experiments.evaluate import evaluate_agent
from rl_engine.config import *

env = OnlineSDNEnv()

agent = DQNAgent(STATE_DIM, ACTION_DIM)
agent.load("models/dqn_model.pth")
agent.epsilon = 0.0  # Đảm bảo agent luôn chọn hành động tốt nhất

agent = PPOAgent(STATE_DIM, ACTION_DIM)
agent.load("models/ppo_model.pth")

state = env.reset()

for step in range(10):
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    print("Action:", action, "Reward:", reward)
    state = next_state