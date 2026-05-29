import os
import torch
import numpy as np
import pandas as pd

from rl_engine.config import STATE_DIM, ACTION_DIM
from rl_engine.offline_env import OfflineSDNEnv
from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.agent.ppo_agent import PPOAgent


ACTION_NAMES = {
    0: "no_action",
    1: "block_suspicious_flow",
    2: "limit_bandwidth",
    3: "redirect_to_honeypot",
    4: "isolate_device",
}


def load_dqn():
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    ckpt = torch.load("models/dqn_model.pth", map_location=agent.device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        agent.q_net.load_state_dict(ckpt["model_state_dict"])

        if "target_model_state_dict" in ckpt:
            agent.target_net.load_state_dict(ckpt["target_model_state_dict"])
        else:
            agent.target_net.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        agent.epsilon = 0.0
    else:
        agent.q_net.load_state_dict(ckpt)
        agent.target_net.load_state_dict(ckpt)
        agent.epsilon = 0.0

    agent.q_net.eval()
    agent.target_net.eval()

    return agent


def load_ppo():
    agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    ckpt = torch.load("models/ppo_model.pth", map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        agent.model.load_state_dict(ckpt["model_state_dict"])
    else:
        agent.model.load_state_dict(ckpt)

    agent.model.eval()
    return agent


def select_dqn_action(agent, state):
    with torch.no_grad():
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = agent.q_net(x)
        return int(torch.argmax(q_values, dim=1).item())


def select_ppo_action(agent, state):
    return int(agent.select_greedy_action(state))


def evaluate(agent, agent_name, data_path):
    df = pd.read_csv(data_path)
    env = OfflineSDNEnv(dataframe=df)

    if hasattr(env, "max_steps_per_episode"):
        env.max_steps_per_episode = len(df)

    state, _ = env.reset()
    done = False

    total_reward = 0.0
    steps = 0

    action_counts = {i: 0 for i in range(ACTION_DIM)}
    reward_by_attack = {}

    while not done:
        row = env.df.iloc[env.idx]
        attack_indicator = float(row["attack_indicator"])

        if agent_name == "dqn":
            action = select_dqn_action(agent, state)
        else:
            action = select_ppo_action(agent, state)

        action_counts[action] += 1

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1

        reward_by_attack.setdefault(attack_indicator, []).append(reward)

        state = next_state

    result = {
        "agent": agent_name,
        "data": data_path,
        "steps": steps,
        "total_reward": total_reward,
        "mean_reward": total_reward / max(steps, 1),
    }

    for action_id, count in action_counts.items():
        result[f"action_{action_id}_{ACTION_NAMES[action_id]}"] = count

    for attack_value, rewards in reward_by_attack.items():
        result[f"reward_attack_{attack_value}"] = float(np.mean(rewards))

    return result


def main():
    os.makedirs("results/evaluation", exist_ok=True)

    rows = []

    if os.path.exists("models/dqn_model.pth"):
        dqn = load_dqn()
        rows.append(evaluate(dqn, "dqn", "data/processed/val_data.csv"))
        rows.append(evaluate(dqn, "dqn", "data/processed/test_data.csv"))

    if os.path.exists("models/ppo_model.pth"):
        ppo = load_ppo()
        rows.append(evaluate(ppo, "ppo", "data/processed/val_data.csv"))
        rows.append(evaluate(ppo, "ppo", "data/processed/test_data.csv"))

    out = pd.DataFrame(rows)
    out.to_csv("results/evaluation/offline_eval_results.csv", index=False)

    print(out.to_string(index=False))
    print("\nSaved to results/evaluation/offline_eval_results.csv")


if __name__ == "__main__":
    main()