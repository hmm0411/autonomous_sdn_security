import os
import random
import logging
import numpy as np
import pandas as pd
import torch
import mlflow
from prometheus_client import start_http_server

from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.offline_env import OfflineSDNEnv
from rl_engine.replay_buffer import ReplayBuffer
from rl_engine.logger import Logger
from rl_engine.config import *
from rl_engine.utils import set_seed

logging.basicConfig(level=logging.INFO)
mlflow.set_tracking_uri("http://34.126.64.185:5000")
mlflow.set_experiment("sdn-rl-dqn")

def run_single_seed_dqn(seed_value, df_train):
    """Huấn luyện DQN với một Seed cụ thể"""
    set_seed(seed_value)
    env = OfflineSDNEnv(dataframe=df_train)
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    epsilon = EPS_START
    seed_reward_history = []
    total_episodes = 2 if os.getenv("CI") == "true" else MAX_EPISODES

    for episode in range(total_episodes):
        state, _ = env.reset(seed=seed_value if episode == 0 else None)
        total_reward = 0
        losses = []

        for step in range(MAX_STEPS):
            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, ACTION_DIM - 1)
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            if len(buffer) > BATCH_SIZE:
                loss = agent.update(buffer.sample(BATCH_SIZE))
                losses.append(loss)

            if done: break

        # Cập nhật epsilon
        epsilon = max(EPS_END, epsilon - (EPS_START - EPS_END) / EPS_DECAY)
        seed_reward_history.append(total_reward)

        if episode % 50 == 0:
            logging.info(f"Seed {seed_value} | Ep {episode} | Reward: {total_reward:.2f} | Eps: {epsilon:.3f}")

    return seed_reward_history

def train_multi_seeds_dqn():
    seeds = [42, 101, 123, 456, 789]
    all_results = []
    df_train = pd.read_csv("../../data/processed/train_data.csv")
    
    with mlflow.start_run(run_name="DQN_MultiSeed_Analysis"):
        mlflow.log_param("algo", "DQN")
        
        for s in seeds:
            logging.info(f"\n--- BẮT ĐẦU DQN SEED: {s} ---")
            history = run_single_seed_dqn(s, df_train)
            all_results.append(history)
            
            # Log kết quả cuối của mỗi seed lên MLflow để tiện so sánh nhanh
            mlflow.log_metric(f"final_reward_seed_{s}", history[-1])

        # 3. Tính toán giá trị trung bình (Mean) và độ lệch chuẩn (Std)
        all_results = np.array(all_results)
        mean_rewards = np.mean(all_results, axis=0)
        std_rewards = np.std(all_results, axis=0)
        
        # 4. Lưu kết quả ra file CSV (Tương tự PPO)
        output_df = pd.DataFrame({
            'episode': range(len(mean_rewards)),
            'mean_reward': mean_rewards,
            'std_reward': std_rewards
        })
        
        os.makedirs("../../results", exist_ok=True)
        csv_path = "../../results/dqn_multi_seed_results.csv"
        output_df.to_csv(csv_path, index=False)
        
        # Log file CSV vào MLflow Artifacts
        mlflow.log_artifact(csv_path)
        logging.info(f"Đã lưu kết quả DQN Multi-seed tại: {csv_path}")

if __name__ == "__main__":
    # Dùng port khác với PPO nếu chạy song song
    start_http_server(9000) 
    train_multi_seeds_dqn()