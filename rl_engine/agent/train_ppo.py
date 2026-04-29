import os
import logging
import collections
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
from prometheus_client import start_http_server, Gauge
import torch
from torch.optim.lr_scheduler import LinearLR

# --- IMPORT TỪ REPO CỦA BẠN ---
from rl_engine.online_env import OnlineSDNEnv
from rl_engine.offline_env import OfflineSDNEnv
from rl_engine.agent.ppo_agent import PPOAgent
from rl_engine.logger import Logger
from rl_engine.utils import set_seed
from rl_engine.config import *
# Bổ sung nếu chưa có trong config: 
# STATE_DIM = 7; ACTION_DIM = 5; WINDOW_SIZE = 50; MAX_EPISODES = 500; MAX_STEPS = 1000;

# DIRECTORY STRUCTURE
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# CẤU HÌNH PROMETHEUS METRICS
# ==========================================
PROM_REWARD = Gauge('episode_reward', 'Phần thưởng thô của Episode', ['agent'])
PROM_REWARD_MEAN = Gauge('episode_reward_mean', 'Phần thưởng trung bình', ['agent'])
PROM_REWARD_STD = Gauge('episode_reward_std', 'Độ lệch chuẩn phần thưởng', ['agent'])
PROM_REWARD_BEST = Gauge('episode_reward_best', 'Kỷ lục phần thưởng tốt nhất', ['agent'])
PROM_LOSS = Gauge('training_loss', 'Loss của mô hình', ['agent'])

# ==========================================
# CẤU HÌNH MLOPS & CI/CD
# ==========================================
IS_CI = os.getenv("CI", "false").lower() == "true"

if not IS_CI:
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("SDN_Autonomous_Security")

def run_single_seed_ppo(seed_value, df_train, parent_run=None):
    """Huấn luyện PPO với một Seed cụ thể"""
    set_seed(seed_value)
    env = OfflineSDNEnv(dataframe=df_train)
    agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    
    logger = Logger(log_dir=os.path.join(BASE_DIR, f"runs/ppo_seed_{seed_value}"))
    
    # Các mảng lưu trữ kết quả cho seed này để xuất CSV
    seed_rewards = []
    seed_losses = []
    seed_lrs = [] # PPO dùng Learning Rate thay vì Epsilon như DQN
    
    recent_rewards = collections.deque(maxlen=WINDOW_SIZE)
    best_reward_so_far = float('-inf')

    total_episodes = 2 if IS_CI else MAX_EPISODES
    scheduler = LinearLR(agent.optimizer, start_factor=1.0, end_factor=0.05, total_iters=total_episodes)

    for episode in range(total_episodes):
        state, _ = env.reset(seed=seed_value if episode == 0 else None)
        
        states, actions, rewards, log_probs, dones = [], [], [], [], []
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)

            episode_reward += reward
            state = next_state
            
            if done: 
                break
                
        # Cập nhật mô hình PPO sau mỗi episode
        metrics = agent.update(states, actions, log_probs, rewards, dones)
        
        policy_loss = metrics.get("policy_loss", 0.0) if metrics else 0.0
        value_loss = metrics.get("value_loss", 0.0) if metrics else 0.0
        total_loss = policy_loss + value_loss

        # Thống kê
        recent_rewards.append(episode_reward)
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
        
        if episode_reward > best_reward_so_far:
            best_reward_so_far = episode_reward

        current_lr = scheduler.get_last_lr()[0]
        
        # Lưu vào mảng xuất CSV
        seed_rewards.append(episode_reward)
        seed_losses.append(total_loss)
        seed_lrs.append(current_lr)

        # Bơm dữ liệu Prometheus
        PROM_REWARD.labels(agent='ppo').set(episode_reward)
        PROM_REWARD_MEAN.labels(agent='ppo').set(float(mean_reward))
        PROM_REWARD_STD.labels(agent='ppo').set(float(std_reward))
        PROM_REWARD_BEST.labels(agent='ppo').set(float(best_reward_so_far))
        PROM_LOSS.labels(agent='ppo').set(float(total_loss))

        # Bơm dữ liệu MLflow
        if not IS_CI and parent_run:
            try:
                mlflow.log_metric(f"reward_raw_seed_{seed_value}", episode_reward, step=episode)
                mlflow.log_metric(f"reward_mean_seed_{seed_value}", float(mean_reward), step=episode)
                mlflow.log_metric(f"reward_std_seed_{seed_value}", float(std_reward), step=episode)
                mlflow.log_metric(f"reward_best_seed_{seed_value}", float(best_reward_so_far), step=episode)
                mlflow.log_metric(f"loss_total_seed_{seed_value}", float(total_loss), step=episode)
                mlflow.log_metric(f"loss_policy_seed_{seed_value}", float(policy_loss), step=episode)
                mlflow.log_metric(f"loss_value_seed_{seed_value}", float(value_loss), step=episode)
            except Exception: pass

        # Log nội bộ
        logger.log_ppo(
            episode, 
            episode_reward, 
            policy_loss, 
            value_loss, 
            metrics.get("entropy", 0.0) if metrics else 0.0, 
            actions
        )

        scheduler.step()
        
        if episode % 20 == 0:
            logging.info(f"PPO | Ep {episode} | R: {episode_reward:.1f} | Mean: {mean_reward:.1f} | Best: {best_reward_so_far:.1f}")

    logger.close()
    
    return {
        "rewards": seed_rewards,
        "losses": seed_losses,
        "lrs": seed_lrs
    }, agent

def train_multi_seeds_ppo():
    """Train PPO with multiple seeds and log all metrics"""
    seeds = [42, 101, 123, 456, 789]
    data_path = "./data/processed/train_data.csv"
    if not os.path.exists(data_path):
        data_path = "../../data/processed/train_data.csv"
        
    if os.path.exists(data_path):
        df_train = pd.read_csv(data_path)
    else:
        logging.error(f"Không tìm thấy file data tại {data_path}.")
        df_train = None
    
    if not IS_CI:
        mlflow.start_run(run_name="PPO_Production_Training")
        mlflow.log_param("algo", "PPO")
        mlflow.log_param("seeds", str(seeds))
    
    best_agent_overall = None
    best_overall_mean = -float('inf')

    if best_agent_overall is not None:
        model_path = os.path.join(RESULTS_DIR, "ppo_model.pth")
        torch.save(
            {
                "model_state_dict": best_agent_overall.model.state_dict(), # type: ignore
                "optimizer_state_dict": best_agent_overall.optimizer.state_dict(), # type: ignore
            },
            model_path
        )
        logging.info(f"Saved best overall PPO model to: {model_path}")

    # Dictionary chứa kết quả tổng hợp
    all_results = {"rewards": [], "losses": [], "lrs": []}
    
    for s in seeds:
        logging.info(f"\n--- BẮT ĐẦU SEED (PPO): {s} ---")
        seed_result, trained_agent = run_single_seed_ppo(s, df_train, parent_run=(not IS_CI))
        
        all_results["rewards"].append(seed_result["rewards"])
        all_results["losses"].append(seed_result["losses"])
        all_results["lrs"].append(seed_result["lrs"])
        
        avg_final = np.mean(seed_result["rewards"][-WINDOW_SIZE:])
        if avg_final > best_overall_mean:
            best_overall_mean = avg_final
            best_agent_overall = trained_agent

    if best_agent_overall is not None:
        model_path = os.path.join(RESULTS_DIR, "models", "ppo_model.pth")

        torch.save(
            {
                "model_state_dict": best_agent_overall.model.state_dict(), # type: ignore
                "optimizer_state_dict": best_agent_overall.optimizer.state_dict(), # type: ignore
            },
            model_path,
        )

        logging.info(f"Saved best overall PPO model to: {model_path}")
            
    # --- TÍNH TOÁN THỐNG KÊ TOÀN CỤC ---
    np_rewards = np.array(all_results["rewards"])
    np_losses = np.array(all_results["losses"])
    
    mean_rewards = np.mean(np_rewards, axis=0)
    std_rewards = np.std(np_rewards, axis=0)
    mean_losses = np.mean(np_losses, axis=0)
    std_losses = np.std(np_losses, axis=0)

    # --- LƯU KẾT QUẢ VÀO CSV ---
    try:
        os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
        
        # 1. Summary statistics
        summary_df = pd.DataFrame({
            'episode': range(len(mean_rewards)),
            'mean_reward': mean_rewards,
            'std_reward': std_rewards,
            'mean_loss': mean_losses,
            'std_loss': std_losses
        })
        summary_path = os.path.join(RESULTS_DIR, "models", "ppo_summary_results.csv")
        summary_df.to_csv(summary_path, index=False)
        if not IS_CI: mlflow.log_artifact(summary_path)

        # 2. Individual seed results
        for i, seed in enumerate(seeds):
            seed_df = pd.DataFrame({
                'episode': range(len(all_results["rewards"][i])),
                'reward': all_results["rewards"][i],
                'loss': all_results["losses"][i],
                'learning_rate': all_results["lrs"][i]
            })
            seed_path = os.path.join(RESULTS_DIR, "models", f"ppo_seed_{seed}_results.csv")
            seed_df.to_csv(seed_path, index=False)
            if not IS_CI: mlflow.log_artifact(seed_path)

        # 3. Detailed aggregated metrics
        metrics_df = pd.DataFrame({
            'seed': seeds,
            'final_reward': [all_results["rewards"][i][-1] for i in range(len(seeds))],
            'best_reward': [max(all_results["rewards"][i]) for i in range(len(seeds))],
            'mean_reward': [np.mean(all_results["rewards"][i]) for i in range(len(seeds))],
            'final_loss': [all_results["losses"][i][-1] for i in range(len(seeds))],
            'mean_loss': [np.mean(all_results["losses"][i]) for i in range(len(seeds))]
        })
        metrics_path = os.path.join(RESULTS_DIR, "models", "ppo_metrics_by_seed.csv")
        metrics_df.to_csv(metrics_path, index=False)
        if not IS_CI: mlflow.log_artifact(metrics_path)

        logging.info("Đã lưu toàn bộ CSV thống kê vào thư mục results/")

    except Exception as e:
        logging.error(f"Lỗi khi lưu CSV: {e}")

    logging.info(f"\n Đã hoàn thành PPO Multi-seed training")
    logging.info(f"Final Mean Reward (Across Seeds): {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")
    logging.info(f"Best Mean Reward (Across Seeds): {np.max(mean_rewards):.2f}")

    # Đăng ký model lên Registry
    if not IS_CI and best_agent_overall is not None:
        try:
            # 1. Bỏ qua cảnh báo policy vì ta biết chắc Agent có thuộc tính này
            mlflow.pytorch.log_model(best_agent_overall.model, "ppo_model") # type: ignore
            
            # 2. Lấy active run an toàn
            current_run = mlflow.active_run()
            if current_run is not None:
                run_id = current_run.info.run_id
                mlflow.register_model(f"runs:/{run_id}/ppo_model", "SDN_PPO_Model")
                logging.info("Đã đăng ký SDN_PPO_Model lên MLflow Registry!")
            else:
                logging.warning("Không có Active Run nào trên MLflow. Bỏ qua bước đăng ký Model.")
                
        except Exception as e:
            logging.error(f"Lỗi đăng ký model: {e}")
            
    if not IS_CI:
        mlflow.end_run()

if __name__ == "__main__":
    PORT = 9001
    start_http_server(PORT)
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"[PPO] Prometheus metrics server started on port {PORT}")
    train_multi_seeds_ppo()