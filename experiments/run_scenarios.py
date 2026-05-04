import os
import pandas as pd
import numpy as np
from rl_engine.offline_env import OfflineSDNEnv
from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.agent.ppo_agent import PPOAgent
from experiments.baseline_rule import BaselineRuleBasedAgent
from rl_engine.config import STATE_DIM, ACTION_DIM

SCENARIOS = {
    "normal": "data/processed/test_normal.csv",
    "ddos": "data/processed/test_ddos.csv",
    "spoofing": "data/processed/test_spoofing.csv",
    "overflow": "data/processed/test_overflow.csv",
    "packet_in": "data/processed/test_packet_in.csv",
    "port_scan": "data/processed/test_port_scan.csv"
}

EPISODES = 20
RESULT_PATH = "results/evaluation_summary.csv"


def evaluate(env, agent):
    rewards = []
    switching = []

    for _ in range(EPISODES):
        state, _ = env.reset()
        done = False
        total = 0
        actions = []

        while not done:
            if hasattr(agent, "predict"):
                a = agent.predict(state)
                a = a[0] if isinstance(a, tuple) else a
            else:
                a = agent.select_action(state)

            next_state, reward, term, trunc, _ = env.step(int(a))
            total += reward
            actions.append(int(a))
            state = next_state
            done = term or trunc

        rewards.append(total)
        switching.append(np.mean(np.array(actions[1:]) != np.array(actions[:-1])))

    return np.mean(rewards), np.std(rewards), np.mean(switching)


def main():

    all_results = []

    dqn = DQNAgent(STATE_DIM, ACTION_DIM)
    dqn.load("models/dqn_model.pth")

    ppo = PPOAgent(STATE_DIM, ACTION_DIM)
    ppo.load("models/ppo_model.pth")

    rule = BaselineRuleBasedAgent()

    for scenario, path in SCENARIOS.items():

        df = pd.read_csv(path)
        env = OfflineSDNEnv(df)

        for name, agent in {
            "DQN": dqn,
            "PPO": ppo,
            "Rule": rule
        }.items():

            mean_r, std_r, mean_s = evaluate(env, agent)

            all_results.append({
                "scenario": scenario,
                "model": name,
                "mean_reward": mean_r,
                "std_reward": std_r,
                "mean_switching": mean_s
            })

    df_out = pd.DataFrame(all_results)
    df_out.to_csv(RESULT_PATH, index=False)

    print("Saved:", RESULT_PATH)


if __name__ == "__main__":
    main()