import logging
import os
import collections
import pandas as pd
import numpy as np
import torch
import mlflow
from prometheus_client import start_http_server

from rl_engine.agent.ppo_agent import PPOAgent
from rl_engine.logger import Logger
from rl_engine.config import *
from rl_engine.offline_env import OfflineSDNEnv
from rl_engine.utils import set_seed
from torch.optim.lr_scheduler import LinearLR

# Cấu hình Logging & MLflow
logging.basicConfig(level=logging.INFO)
mlflow.set_tracking_uri("http://34.126.64.185:5000")
mlflow.set_experiment("sdn-rl-ppo")

def run_single_seed(seed_value, df_train, parent_run=None):
    """Huấn luyện Agent với một Seed cụ thể"""
    set_seed(seed_value)
    env = OfflineSDNEnv(dataframe=df_train)
    agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    
    seed_reward_history = []
    best_avg_reward = -float('inf')
    recent_rewards = collections.deque(maxlen=10)
    
    total_episodes = 2 if os.getenv("CI") == "true" else MAX_EPISODES
    scheduler = LinearLR(agent.optimizer, start_factor=1.0, end_factor=0.05, total_iters=total_episodes)

    # Khởi tạo Logger riêng cho từng seed nếu cần (hoặc dùng chung)
    logger = Logger(log_dir=f"../../runs/ppo_seed_{seed_value}")

    for episode in range(total_episodes):
        state, _ = env.reset(seed=seed_value if episode == 0 else None)
        
        # Buffer tạm thời để chứa quỹ đạo (Trajectory) của 1 episode
        states, actions, rewards, log_probs, dones = [], [], [], [], []
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Lưu dữ liệu vào buffer để update
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done or truncated)

            episode_reward += reward
            state = next_state
            if done or truncated: break
            
        # --- QUAN TRỌNG: CẬP NHẬT AGENT (Mất dòng này ở code cũ) ---
        metrics = agent.update(states, actions, log_probs, rewards, dones)
        
        # Lưu lịch sử
        seed_reward_history.append(episode_reward)
        recent_rewards.append(episode_reward)
        
        # Log metrics (MLflow lồng trong Parent Run)
        if parent_run:
            mlflow.log_metric(f"reward_seed_{seed_value}", episode_reward, step=episode)

        # Cơ chế lưu Best Model
        if len(recent_rewards) == 10:
            avg_reward = np.mean(recent_rewards)
            if avg_reward > best_avg_reward and episode > 20:
                best_avg_reward = avg_reward
                save_path = f"../../models/best_ppo_seed_{seed_value}.pth"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                agent.save(save_path) if hasattr(agent, "save") else torch.save(agent.policy.state_dict(), save_path)
                logging.info(f"[Seed {seed_value}] Ep {episode}: New Best Avg Reward: {best_avg_reward:.2f}")

        scheduler.step()
        
        if episode % 50 == 0:
            logging.info(f"Seed {seed_value} | Ep {episode} | Reward: {episode_reward:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    logger.close()
    return seed_reward_history

def train_multi_seeds():
    seeds = [42, 101, 123] # Rút ngắn để test, tăng lên khi chạy thật
    all_results = []
    df_train = pd.read_csv("../../data/processed/train_data.csv")
    
    with mlflow.start_run(run_name="PPO_MultiSeed_Analysis"):
        mlflow.log_param("algo", "PPO")
        mlflow.log_param("seeds", str(seeds))
        
        for s in seeds:
            logging.info(f"\n--- BẮT ĐẦU SEED: {s} ---")
            history = run_single_seed(s, df_train, parent_run=True)
            all_results.append(history)
        
        # Tính Mean & Std để vẽ biểu đồ bóng mờ (Shaded line plot)
        all_results = np.array(all_results)
        mean_rewards = np.mean(all_results, axis=0)
        std_rewards = np.std(all_results, axis=0)
        
        # Lưu kết quả
        output_df = pd.DataFrame({
            'episode': range(len(mean_rewards)),
            'mean_reward': mean_rewards,
            'std_reward': std_rewards
        })
        res_path = "../../results/ppo_multi_seed_results.csv"
        os.makedirs("../../results", exist_ok=True)
        output_df.to_csv(res_path, index=False)
        mlflow.log_artifact(res_path)
        logging.info(f"Quá trình huấn luyện hoàn tất. Kết quả lưu tại {res_path}")

if __name__ == "__main__":
    port = 9001 # Tránh trùng với DQN nếu chạy song song
    start_http_server(port)
    train_multi_seeds()