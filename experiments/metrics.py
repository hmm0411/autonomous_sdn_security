import importlib
import numpy as np
import pandas as pd
import torch
from rl_engine.offline_env import OfflineSDNEnv
from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.agent.ppo_agent import PPOAgent
from rl_engine.config import *


def _load_rule_agent_class():
    candidates = [
        ("rl_engine.baseline_rule", "BaselineRuleBasedAgent"),
        ("rl_engine.baseline_rule_based", "BaselineRuleBasedAgent"),
        ("rl_engine.baseline", "BaselineRuleBasedAgent"),
    ]

    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue

    class BaselineRuleBasedAgent:
        def predict(self, state):
            return 0

        def select_action(self, state):
            return 0

    return BaselineRuleBasedAgent


def evaluate_agent(env, agent, max_steps=1000):

    state, _ = env.reset()
    done = False

    total_reward = 0
    action_history = []
    rewards = []

    steps = 0

    while not done and steps < max_steps:

        if hasattr(agent, "predict"):
            action = agent.predict(state)
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        rewards.append(reward)
        action_history.append(action)

        state = next_state
        done = terminated or truncated
        steps += 1

    switching_rate = np.mean(
        np.array(action_history[1:]) != np.array(action_history[:-1])
    )

    return {
        "total_reward": total_reward,
        "avg_reward": np.mean(rewards),
        "switching_rate": switching_rate,
        "actions": action_history,
    }


def main():

    df_test = pd.read_csv("dataset_test.csv")
    env = OfflineSDNEnv(dataframe=df_test)

    print("Evaluating DQN...")
    dqn = DQNAgent(STATE_DIM, ACTION_DIM)
    dqn.q_net.load_state_dict(torch.load("dqn_model.pth"))
    dqn_result = evaluate_agent(env, dqn)

    print("Evaluating PPO...")
    ppo = PPOAgent(STATE_DIM, ACTION_DIM)
    ppo.load("ppo_model.pth")
    ppo_result = evaluate_agent(env, ppo)

    print("Evaluating Rule-based...")
    RuleAgentClass = _load_rule_agent_class()
    rule = RuleAgentClass()
    rule_result = evaluate_agent(env, rule)

    print("\n===== RESULTS =====")
    print("DQN:", dqn_result)
    print("PPO:", ppo_result)
    print("Rule:", rule_result)


if __name__ == "__main__":
    main()