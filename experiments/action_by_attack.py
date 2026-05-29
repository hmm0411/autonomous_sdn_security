import os
import torch
import pandas as pd
from collections import defaultdict

from rl_engine.config import STATE_DIM, ACTION_DIM
from rl_engine.offline_env import OfflineSDNEnv
from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.agent.ppo_agent import PPOAgent


ATTACK_NAMES = {
    0.0: "normal",
    0.2: "ddos",
    0.4: "flow_overflow",
    0.6: "ip_spoofing",
    0.8: "packet_in_flood",
    1.0: "port_scanning",
}

ACTION_NAMES = {
    0: "no_action",
    1: "block",
    2: "limit",
    3: "redirect",
    4: "isolate",
}


def load_dqn():
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    ckpt = torch.load("models/dqn_model.pth", map_location=agent.device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        agent.q_net.load_state_dict(ckpt["model_state_dict"])
    else:
        agent.q_net.load_state_dict(ckpt)

    agent.epsilon = 0.0
    agent.q_net.eval()
    return agent


def dqn_action(agent, state):
    with torch.no_grad():
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        q = agent.q_net(x)
        return int(torch.argmax(q, dim=1).item())


def evaluate_dqn_by_attack(data_path):
    df = pd.read_csv(data_path)
    env = OfflineSDNEnv(dataframe=df, max_steps_per_episode=len(df))

    state, _ = env.reset(seed=42)

    table = defaultdict(lambda: defaultdict(int))
    rewards = defaultdict(list)

    done = False
    agent = load_dqn()

    while not done:
        row = env.df.iloc[env.idx]
        attack = float(row["attack_indicator"])

        action = dqn_action(agent, state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        table[attack][action] += 1
        rewards[attack].append(reward)

        state = next_state

    rows = []

    for attack, action_counts in sorted(table.items()):
        row = {
            "attack": ATTACK_NAMES.get(attack, str(attack)),
            "mean_reward": sum(rewards[attack]) / len(rewards[attack]),
        }

        for action_id, action_name in ACTION_NAMES.items():
            row[action_name] = action_counts[action_id]

        rows.append(row)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    os.makedirs("results/evaluation", exist_ok=True)
    out.to_csv("results/evaluation/dqn_action_by_attack.csv", index=False)


if __name__ == "__main__":
    evaluate_dqn_by_attack("data/processed/test_data.csv")