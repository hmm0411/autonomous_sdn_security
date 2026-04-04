import os
import random
import numpy as np
import pandas as pd
import torch
import torch

from rl_engine import env
from rl_engine.data_processor import process_sdn_dataset
from rl_engine.env import SDNEnv
from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.offline_env import OfflineSDNEnv
from rl_engine.replay_buffer import ReplayBuffer
from rl_engine.logger import Logger
from rl_engine.config import *

def train():
    df = pd.read_csv("../../data/processed/train_data.csv")
    env = OfflineSDNEnv(dataframe=df)

    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM
    )

    buffer = ReplayBuffer(BUFFER_SIZE)

    logger = Logger()

    epsilon = EPS_START

    total_episodes = 2 if os.getenv("CI") == "true" else MAX_EPISODES

    for episode in range(total_episodes):

        state, _ = env.reset()

        total_reward = 0
        action_history = []
        losses = []

        for step in range(MAX_STEPS):

            # epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, ACTION_DIM - 1)
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add((state, action, reward, next_state, done))

            action_history.append(action)

            total_reward += reward

            state = next_state

            if len(buffer) > BATCH_SIZE:

                batch = buffer.sample(BATCH_SIZE)

                loss = agent.update(batch)

                losses.append(loss)

            if done:
                break

        epsilon = max(
            EPS_END,
            epsilon - (EPS_START - EPS_END) / EPS_DECAY
        )

        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        logger.log_dqn(
            episode,
            total_reward,
            avg_loss,
            epsilon,
            action_history
        )

        print(
            f"Episode {episode} | "
            f"Reward: {total_reward:.3f} | "
            f"Loss: {avg_loss:.4f} | "
            f"Epsilon: {epsilon:.3f}"
        )

    logger.save_dqn()

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    save_path = os.path.join(MODELS_DIR, "dqn_model.pth")
    torch.save(agent.q_net.state_dict(), save_path)
    print(f"Đã lưu model DQN tại: {save_path}")


if __name__ == "__main__":
    train()