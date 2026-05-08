import numpy as np
from rl_engine.online_env import OnlineSDNEnv
from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.config import STATE_DIM, ACTION_DIM

def run_robustness_test():

    env = OnlineSDNEnv()
    agent = DQNAgent(STATE_DIM, ACTION_DIM)
    agent.load("models/dqn_model.pth")
    agent.epsilon = 0.0

    state = env.reset()

    print("\n=== ROBUSTNESS TEST ===")

    for step in range(30):

        # Inject synthetic spike
        if step == 10:
            env.inject_traffic_spike()

        if step == 20:
            env.inject_attack_pattern()

        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        print(f"Step {step} | Action {action} | Reward {reward}")

        state = next_state