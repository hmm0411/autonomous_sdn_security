import os

import pandas as pd

import numpy as np
import torch
import torch

from rl_engine.env import SDNEnv
from rl_engine.data_processor import process_sdn_dataset
from rl_engine.agent.ppo_agent import PPOAgent
from rl_engine.logger import Logger
from rl_engine.config import *
from rl_engine.offline_env import OfflineSDNEnv

def train():
    df = pd.read_csv("../../data/processed/train_data.csv")
    env = OfflineSDNEnv(dataframe=df)
    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM
    )

    logger = Logger()

    total_episodes = 2 if os.getenv("CI") == "true" else MAX_EPISODES   

    for episode in range(total_episodes):

        state, _ = env.reset()

        states = []
        actions = []
        rewards = []
        log_probs = []
        dones = []

        action_history = []

        total_reward = 0

        for step in range(MAX_STEPS):

            action, log_prob = agent.select_action(state)

            next_state, reward, done, truncated, info = env.step(action)

            done = done or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)

            action_history.append(action)

            total_reward += reward

            state = next_state

            if done:
                break

        metrics = agent.update(
            states,
            actions,
            log_probs,
            rewards,
            dones
        )

        logger.log_ppo(
            episode,
            total_reward,
            metrics["policy_loss"],
            metrics["value_loss"],
            metrics["entropy"],
            action_history
        )

        print(
            f"Episode {episode} | "
            f"Reward: {total_reward:.3f} | "
            f"PolicyLoss: {metrics['policy_loss']:.4f} | "
            f"ValueLoss: {metrics['value_loss']:.4f} | "
            f"Entropy: {metrics['entropy']:.4f}"
        )

    logger.save_ppo()

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    save_path = os.path.join(MODELS_DIR, "ppo_model.pth")
    if hasattr(agent, "save"):
        agent.save(save_path)
    else:
        torch.save(agent.policy.state_dict(), save_path)
    print(f"Đã lưu model PPO tại: {save_path}")

if __name__ == "__main__":
    train()