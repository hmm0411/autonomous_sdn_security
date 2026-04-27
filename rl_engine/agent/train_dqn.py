import os
import random
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
from prometheus_client import start_http_server, Gauge
import collections

# Giả sử các module này đã được định nghĩa đúng trong source code của bạn
# from rl_engine.agent.train_ppo import run_single_seed, train_multi_seeds # (Bỏ dòng này nếu không cần thiết trong file DQN)
from rl_engine.env import SDNEnv
from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.offline_env import OfflineSDNEnv
from rl_engine.replay_buffer import ReplayBuffer
from rl_engine.logger import Logger
from rl_engine.config import *
from rl_engine.utils import set_seed

# CẤU HÌNH PROMETHEUS METRICS
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

def run_single_seed_dqn(seed_value, df_train, parent_run=None):
    """Huấn luyện DQN với một Seed cụ thể"""
    set_seed(seed_value)
    env = OfflineSDNEnv(dataframe=df_train)
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    logger = Logger(log_dir=f"../../runs/dqn_seed_{seed_value}")
    
    epsilon = EPS_START
    
    # Các mảng lưu trữ cho seed này
    seed_rewards = []
    seed_losses = []
    seed_epsilons = []
    
    recent_rewards = collections.deque(maxlen=WINDOW_SIZE)
    best_reward_so_far = float('-inf')

    total_episodes = 2 if IS_CI else MAX_EPISODES

    run_context = mlflow.start_run(run_name=f"Seed_{seed_value}", nested=True) if (not IS_CI and parent_run) else None

    try:
        for episode in range(total_episodes):
            state, _ = env.reset(seed=seed_value if episode == 0 else None)
            episode_reward = 0
            episode_losses = []
            actions_in_episode = []

            for step in range(MAX_STEPS):
                # Epsilon-greedy
                if random.random() < epsilon:
                    action = random.randint(0, ACTION_DIM - 1)
                else:
                    action = agent.select_action(state)
                
                actions_in_episode.append(action)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                buffer.add((state, action, reward, next_state, done))

                # Train Agent
                if len(buffer) > BATCH_SIZE:
                    loss = agent.update(buffer.sample(BATCH_SIZE))
                    if loss is not None:
                        episode_losses.append(loss)

                state = next_state
                episode_reward += reward

                if done:
                    break

            # Cập nhật epsilon
            epsilon = max(EPS_END, epsilon - (EPS_START - EPS_END) / EPS_DECAY)
        
            # Thống kê cho episode
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        
            seed_rewards.append(episode_reward)
            seed_losses.append(avg_loss)
            seed_epsilons.append(epsilon)
        
            recent_rewards.append(episode_reward)
        
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
        
            if episode_reward > best_reward_so_far:
                best_reward_so_far = episode_reward

            # Log nội bộ
            logger.log_dqn(
                episode=episode,
                reward=episode_reward,
                loss=avg_loss,
                epsilon=epsilon,
                actions=actions_in_episode
            )

            # Bơm dữ liệu Prometheus
            PROM_REWARD.labels(agent='dqn').set(episode_reward)
            PROM_REWARD_MEAN.labels(agent='dqn').set(float(mean_reward))
            PROM_REWARD_STD.labels(agent='dqn').set(float(std_reward))
            PROM_REWARD_BEST.labels(agent='dqn').set(float(best_reward_so_far))
            PROM_LOSS.labels(agent='dqn').set(float(avg_loss))
        
            # Bơm dữ liệu MLflow
            if not IS_CI and parent_run:
                try:
                    mlflow.log_metric(f"reward_raw_seed_{seed_value}", float(episode_reward), step=episode)
                    mlflow.log_metric(f"reward_mean_seed_{seed_value}", float(mean_reward), step=episode)
                    mlflow.log_metric(f"reward_std_seed_{seed_value}", float(std_reward), step=episode)
                    mlflow.log_metric(f"reward_best_seed_{seed_value}", float(best_reward_so_far), step=episode)
                    mlflow.log_metric(f"loss_seed_{seed_value}", float(avg_loss), step=episode)
                except Exception: pass
                
            if episode % 20 == 0:
                logging.info(f"DQN | Ep {episode} | R: {episode_reward:.1f} | Mean: {mean_reward:.1f} | Best: {best_reward_so_far:.1f}")

    finally:
        # Đảm bảo luôn đóng Child Run
        if run_context:
            mlflow.end_run()

    return {
        "rewards": seed_rewards,
        "losses": seed_losses,
        "epsilons": seed_epsilons
    }, agent
    

def train_multi_seeds_dqn():
    """Train DQN with multiple seeds and log all metrics"""
    seeds = [42, 123, 456]
    data_path = "./data/processed/train_data.csv"
    if not os.path.exists(data_path):
        data_path = "../../data/processed/train_data.csv"
    
    # Kiểm tra xem file có tồn tại không trước khi đọc
    if os.path.exists(data_path):
        df_train = pd.read_csv(data_path)
    else:
        raise FileNotFoundError(f"Data file not found at {data_path}. Please ensure the training data is available.")

    if not IS_CI:
        mlflow.start_run(run_name="DQN_Production_Training")
        mlflow.log_param("algo", "DQN")
        mlflow.log_param("seeds", str(seeds))
    
    best_agent_overall = None
    best_overall_mean = -float('inf')
    
    # Dictionary chứa kết quả tổng hợp
    all_results = {"rewards": [], "losses": [], "epsilons": []}

    for s in seeds:
        logging.info(f"\n--- BẮT ĐẦU SEED (DQN): {s} ---")
        seed_result, trained_agent = run_single_seed_dqn(s, df_train, parent_run=(not IS_CI))
        
        all_results["rewards"].append(seed_result["rewards"])
        all_results["losses"].append(seed_result["losses"])
        all_results["epsilons"].append(seed_result["epsilons"])
        
        # So sánh dựa trên trung bình 50 tập cuối
        avg_final = np.mean(seed_result["rewards"][-WINDOW_SIZE:])
        if avg_final > best_overall_mean:
            best_overall_mean = avg_final
            best_agent_overall = trained_agent

    # --- TÍNH TOÁN THỐNG KÊ TOÀN CỤC ---
    # Chuyển list of lists thành Numpy array để dễ tính mean/std theo cột
    np_rewards = np.array(all_results["rewards"])
    np_losses = np.array(all_results["losses"])
    
    mean_rewards = np.mean(np_rewards, axis=0)
    std_rewards = np.std(np_rewards, axis=0)
    mean_losses = np.mean(np_losses, axis=0)
    std_losses = np.std(np_losses, axis=0)

    # --- LƯU KẾT QUẢ VÀO CSV ---
    try:
        os.makedirs("../../results", exist_ok=True)
        
        # 1. Summary statistics
        summary_df = pd.DataFrame({
            'episode': range(len(mean_rewards)),
            'mean_reward': mean_rewards,
            'std_reward': std_rewards,
            'mean_loss': mean_losses,
            'std_loss': std_losses
        })
        summary_path = "../../results/dqn_summary_results.csv"
        summary_df.to_csv(summary_path, index=False)
        if not IS_CI: mlflow.log_artifact(summary_path)
        logging.info(f"Saved summary results to: {summary_path}")

        # 2. Individual seed results
        for i, seed in enumerate(seeds):
            seed_df = pd.DataFrame({
                'episode': range(len(all_results["rewards"][i])),
                'reward': all_results["rewards"][i],
                'loss': all_results["losses"][i],
                'epsilon': all_results["epsilons"][i]
            })
            seed_path = f"../../results/dqn_seed_{seed}_results.csv"
            seed_df.to_csv(seed_path, index=False)
            if not IS_CI: mlflow.log_artifact(seed_path)
        logging.info(f"Saved individual seed results to ../../results/dqn_seed_*.csv")

        # 3. Detailed aggregated metrics
        metrics_df = pd.DataFrame({
            'seed': seeds,
            'final_reward': [all_results["rewards"][i][-1] for i in range(len(seeds))],
            'best_reward': [max(all_results["rewards"][i]) for i in range(len(seeds))],
            'mean_reward': [np.mean(all_results["rewards"][i]) for i in range(len(seeds))],
            'final_loss': [all_results["losses"][i][-1] for i in range(len(seeds))],
            'mean_loss': [np.mean(all_results["losses"][i]) for i in range(len(seeds))]
        })
        metrics_path = "../../results/dqn_metrics_by_seed.csv"
        metrics_df.to_csv(metrics_path, index=False)
        if not IS_CI: mlflow.log_artifact(metrics_path)
        logging.info(f"Saved detailed metrics to: {metrics_path}")

    except Exception as e:
        logging.error(f"Failed to save results: {e}")

    logging.info(f"\nĐã hoàn thành DQN Multi-seed training")
    logging.info(f"Final Mean Reward (Across Seeds): {float(mean_rewards[-1]):.2f} ± {float(std_rewards[-1]):.2f}")
    logging.info(f"Best Mean Reward (Across Seeds): {float(np.max(mean_rewards)):.2f}")

    # Đăng ký model lên Registry
    if not IS_CI and best_agent_overall is not None:
        try:
             mlflow.pytorch.log_model(best_agent_overall.q_network, "dqn_model") # type: ignore
             current_run = mlflow.active_run()
             if current_run is not None:
                 run_id = current_run.info.run_id
                 mlflow.register_model(f"runs:/{run_id}/dqn_model", "SDN_DQN_Model")
                 logging.info("Đã đăng ký SDN_DQN_Model lên Registry!")
             else:
                 logging.warning("Không có Active Run nào trên MLflow. Bỏ qua bước đăng ký Model.")
        except Exception as e:
             logging.error(f"Failed to register model: {e}")
             
    if not IS_CI:
        mlflow.end_run()


if __name__ == "__main__":
    start_http_server(9000)
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"[DQN] Prometheus metrics server started on port 9000")
    train_multi_seeds_dqn()