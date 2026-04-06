import os
import collections
import pandas as pd
import random
import numpy as np
import torch
import torch

from rl_engine.env import SDNEnv    
from rl_engine.data_processor import process_sdn_dataset
from rl_engine.agent.ppo_agent import PPOAgent
from rl_engine.logger import Logger
from rl_engine.config import *
from rl_engine.offline_env import OfflineSDNEnv
from rl_engine.utils import set_seed
from torch.optim.lr_scheduler import LinearLR

    
def train():
    SEED = 42
    set_seed(SEED)
    df = pd.read_csv("../../data/processed/train_data.csv")
    env = OfflineSDNEnv(dataframe=df)
    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM
    )

    log_dir = f"../../runs/ppo_seed_{SEED}"
    logger = Logger(log_dir=log_dir)

    total_episodes = 2 if os.getenv("CI") == "true" else MAX_EPISODES   

    for episode in range(total_episodes):

        state, _ = env.reset(seed=SEED if episode == 0 else None)  # Chỉ set seed cho episode đầu tiên để đảm bảo tính nhất quán trong CI

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
            
        for _ in range(3):  # PPO thường cập nhật nhiều lần trên cùng một batch - K-epochs để agent học lại batch dữ liệu đó 3 lần trước khi đi tiếp (PPO tái sử dụng dữ liệu)
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

    logger.close()

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    save_path = os.path.join(MODELS_DIR, "ppo_model.pth")
    if hasattr(agent, "save"):
        agent.save(save_path)
    else:
        torch.save(agent.policy.state_dict(), save_path)
    print(f"Đã lưu model PPO tại: {save_path}")

def run_single_seed(seed_value, df_train):
    """Huấn luyện Agent với một Seed cụ thể và trả về lịch sử Reward"""
    set_seed(seed_value)
    
    # Khởi tạo lại toàn bộ cho mỗi Seed mới
    env = OfflineSDNEnv(dataframe=df_train)
    agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    
    seed_reward_history = []

    # KHỞI TẠO CÁC BIẾN THEO DÕI BEST MODEL 
    best_avg_reward = -float('inf')  # Khởi tạo bằng âm vô cực
    # Lưu tối đa 10 phần thưởng gần nhất để tính trung bình
    recent_rewards = collections.deque(maxlen=10)
    
    total_episodes = 2 if os.getenv("CI") == "true" else MAX_EPISODES

    # Giảm tuyến tính LR từ 100% (start_factor=1.0) xuống còn 5% (end_factor=0.05) vào cuối quá trình
    scheduler = LinearLR(agent.optimizer, start_factor=1.0, end_factor=0.05, total_iters=total_episodes)
    
    for episode in range(total_episodes):
        state, _ = env.reset(seed=seed_value if episode == 0 else None)
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            if done or truncated: break
            
        # Update agent sau mỗi episode (hoặc theo batch tùy code của bạn)
        # agent.update(...) 
        
        seed_reward_history.append(episode_reward)
        recent_rewards.append(episode_reward) # Thêm vào cửa sổ trượt
        if len(recent_rewards) == 10 and episode > 50:
            avg_reward = np.mean(recent_rewards)
            
            # Nếu phá kỷ lục trung bình -> Lưu ngay lập tức!
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                
                # Lưu file theo tên seed để không ghi đè lẫn nhau
                save_path = f"../../models/best_ppo_seed_{seed_value}.pth"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Tùy hàm save của bạn, ví dụ:
                if hasattr(agent, "save"):
                    agent.save(save_path)
                    agent.save("../../models/ppo_model.pth")
                else:
                    torch.save(agent.policy.state_dict(), save_path)
                    torch.save(agent.policy.state_dict(), "../../models/ppo_model.pth")
                
                print(f"[Seed {seed_value}] Ep {episode}: Lưu Best Model mới với Avg Reward = {best_avg_reward:.2f}")
        
        scheduler.step()  # Cập nhật LR sau mỗi episode
        
        if episode % 50 == 0:
            print(f"Seed {seed_value} | Ep {episode} | Reward: {episode_reward:.2f}")

        if os.getenv("CI") == "true":
            save_path = "../../models/ppo_model.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if hasattr(agent, "save"):
                agent.save(save_path)
            else:
                torch.save(agent.policy.state_dict(), save_path)    
    return seed_reward_history

def train_multi_seeds():
    # 1. Danh sách các Seeds để thử nghiệm
    seeds = [42, 101, 123, 456, 789] 
    all_results = []
    
    df_train = pd.read_csv("../../data/processed/train_data.csv")
    
    # 2. Chạy huấn luyện cho từng Seed
    for s in seeds:
        print(f"\nBẮT ĐẦU HUẤN LUYỆN VỚI SEED: {s}")
        history = run_single_seed(s, df_train)
        all_results.append(history)
    
    # 3. Tính toán giá trị trung bình (Mean) và độ lệch chuẩn (Std)
    all_results = np.array(all_results) # Shape: (Số_Seed, Số_Episodes)
    mean_rewards = np.mean(all_results, axis=0)
    std_rewards = np.std(all_results, axis=0)
    
    # 4. Lưu kết quả ra file CSV để vẽ biểu đồ xịn bằng Matplotlib/Seaborn
    output_df = pd.DataFrame({
        'episode': range(len(mean_rewards)),
        'mean_reward': mean_rewards,
        'std_reward': std_rewards
    })
    os.makedirs("../../results", exist_ok=True)
    output_df.to_csv("../../results/ppo_multi_seed_results.csv", index=False)
    print("\nĐã lưu kết quả đa hạt giống vào thư mục results/")

if __name__ == "__main__":
    train_multi_seeds()