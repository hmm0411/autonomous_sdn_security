import os
import random
import logging
import collections
import traceback
import numpy as np
import pandas as pd
import torch

import mlflow
from mlflow.tracking import MlflowClient
from prometheus_client import start_http_server, Gauge
from dotenv import load_dotenv
from mlflow.pytorch import log_model as mlflow_pytorch_log_model

# Import từ source code của bạn
from rl_engine.online_env import OnlineSDNEnv
from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.offline_env import OfflineSDNEnv
from rl_engine.replay_buffer import ReplayBuffer
from rl_engine.logger import Logger
from rl_engine.config import *
from rl_engine.utils import set_seed
from mlops.mlflow_manager import get_best_model

load_dotenv()

# ==========================================
# CẤU HÌNH PROMETHEUS METRICS
# ==========================================
PROM_REWARD = Gauge('episode_reward', 'Phần thưởng thô của Episode', ['agent'])
PROM_REWARD_MEAN = Gauge('episode_reward_mean', 'Phần thưởng trung bình', ['agent'])
PROM_REWARD_STD = Gauge('episode_reward_std', 'Độ lệch chuẩn phần thưởng', ['agent'])
PROM_REWARD_BEST = Gauge('episode_reward_best', 'Kỷ lục phần thưởng tốt nhất', ['agent'])
PROM_LOSS = Gauge('training_loss', 'Loss của mô hình', ['agent'])

# ==========================================
# CẤU HÌNH DIRECTORY STRUCTURE
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# ==========================================
# CẤU HÌNH MLOPS & CI/CD
# ==========================================
IS_CI = os.getenv("CI", "false").lower() == "true"
client = MlflowClient()

if not IS_CI:
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ROOT_USER", "minioadmin")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9005")
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("SDN_Autonomous_Security")

def run_single_seed_dqn(seed_value, df_train, parent_run=False):
    """Huấn luyện DQN với một Seed cụ thể"""
    set_seed(seed_value)
    env = OfflineSDNEnv(dataframe=df_train)
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    logger = Logger(log_dir=os.path.join(RUNS_DIR, f"dqn_seed_{seed_value}"))
    epsilon = EPS_START
    
    seed_rewards, seed_losses, seed_epsilons = [], [], []
    recent_rewards = collections.deque(maxlen=WINDOW_SIZE)
    best_reward_so_far = float('-inf')
    total_episodes = 2 if IS_CI else MAX_EPISODES

    # Khởi tạo nested run nếu không phải môi trường CI
    active_run = mlflow.start_run(run_name=f"Seed_{seed_value}", nested=True) if (not IS_CI and parent_run) else None

    try:
        for episode in range(total_episodes):
            state, _ = env.reset(seed=seed_value if episode == 0 else None)
            episode_reward = 0
            episode_losses = []
            actions_in_episode = []

            for step in range(MAX_STEPS):
                if random.random() < epsilon:
                    action = random.randint(0, ACTION_DIM - 1)
                else:
                    action = agent.select_action(state)
                
                actions_in_episode.append(action)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                buffer.add((state, action, reward, next_state, done))

                if len(buffer) > BATCH_SIZE:
                    loss = agent.update(buffer.sample(BATCH_SIZE))
                    if loss is not None:
                        episode_losses.append(loss)

                state = next_state
                episode_reward += reward

                if done:
                    break

            epsilon = max(EPS_END, epsilon - (EPS_START - EPS_END) / EPS_DECAY)
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            
            seed_rewards.append(episode_reward)
            seed_losses.append(avg_loss)
            seed_epsilons.append(epsilon)
            recent_rewards.append(episode_reward)
            
            mean_reward = float(np.mean(recent_rewards))
            std_reward = float(np.std(recent_rewards)) if len(recent_rewards) > 1 else 0.0
            
            if episode_reward > best_reward_so_far:
                best_reward_so_far = episode_reward

            logger.log_dqn(
                episode=episode,
                reward=episode_reward,
                loss=avg_loss,
                epsilon=epsilon,
                actions=actions_in_episode
            )

            # Prometheus
            PROM_REWARD.labels(agent='dqn').set(episode_reward)
            PROM_REWARD_MEAN.labels(agent='dqn').set(mean_reward)
            PROM_REWARD_STD.labels(agent='dqn').set(std_reward)
            PROM_REWARD_BEST.labels(agent='dqn').set(float(best_reward_so_far))
            PROM_LOSS.labels(agent='dqn').set(avg_loss)
            
            # MLflow Logging
            if not IS_CI and parent_run:
                try:
                    # Log metric vào child run thay vì parent run
                    mlflow.log_metric("reward_raw", float(episode_reward), step=episode)
                    mlflow.log_metric("reward_mean", mean_reward, step=episode)
                    mlflow.log_metric("reward_std", std_reward, step=episode)
                    mlflow.log_metric("reward_best", float(best_reward_so_far), step=episode)
                    mlflow.log_metric("loss", avg_loss, step=episode)
                except Exception as e: 
                    logging.error(f"Lỗi MLflow ở DQN Seed {seed_value}: {e}")
            
            if episode % 20 == 0:
                logging.info(f"DQN Seed {seed_value} | Ep {episode} | R: {episode_reward:.1f} | Mean: {mean_reward:.1f} | Best: {best_reward_so_far:.1f}")

    finally:
        if active_run:
            mlflow.end_run()
            
    agent.epsilon = epsilon

    return {
        "rewards": seed_rewards,
        "losses": seed_losses,
        "epsilons": seed_epsilons
    }, agent


def train_multi_seeds_dqn():
    """Train DQN with multiple seeds and log all metrics"""
    seeds = [42, 101, 123, 456, 789]
    
    # Tìm kiếm file data thông minh hơn
    possible_paths = ["./data/processed/train_data.csv", "../../data/processed/train_data.csv"]
    data_path = next((path for path in possible_paths if os.path.exists(path)), None)
    
    if data_path is None:
        raise FileNotFoundError("Không tìm thấy file train_data.csv. Vui lòng kiểm tra lại đường dẫn.")
        
    df_train = pd.read_csv(data_path)

    if not IS_CI:
        mlflow.start_run(run_name="DQN_Production_Training")
        mlflow.log_params({
            "algo": "DQN",
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "gamma": 0.95,
            "lr": 1e-4,
            "batch_size": BATCH_SIZE, 
            "episodes": MAX_EPISODES  
        })
        mlflow.log_param("seeds", str(seeds))
        mlflow.set_tags({"project": "NT548", "type": "RL", "env": "production", "author": "Ha My Nguyen"})

    best_agent_overall = None
    best_overall_mean = -float('inf')
    trained_agent = None  # FIX: Khởi tạo biến để tránh UnboundLocalError
    
    all_results = {"rewards": [], "losses": [], "epsilons": []}

    for s in seeds:
        logging.info(f"\n--- BẮT ĐẦU SEED (DQN): {s} ---")
        seed_result, trained_agent = run_single_seed_dqn(s, df_train, parent_run=True)
        
        all_results["rewards"].append(seed_result["rewards"])
        all_results["losses"].append(seed_result["losses"])
        all_results["epsilons"].append(seed_result["epsilons"])
        
        avg_final = float(np.mean(seed_result["rewards"][-WINDOW_SIZE:]))
        if avg_final > best_overall_mean:
            best_overall_mean = avg_final
            best_agent_overall = trained_agent

    # Fallback nếu không có seed nào update được best agent
    if best_agent_overall is None:
        if trained_agent is None:
            raise ValueError("Không có agent nào được huấn luyện. Vui lòng kiểm tra lại cấu hình seeds.")
        logging.warning("No best agent found. Using the last trained agent as fallback.")
        best_agent_overall = trained_agent  # FIX: Sử dụng agent cuối cùng làm fallback thay vì gán None

    model_path = os.path.join(MODELS_DIR, "dqn_model.pth")
    
    torch.save({
        "model_state_dict": best_agent_overall.q_net.state_dict(),
        "target_model_state_dict": best_agent_overall.target_net.state_dict(),
        "optimizer_state_dict": best_agent_overall.optimizer.state_dict(),
        "epsilon": best_agent_overall.epsilon,
    }, model_path)
    logging.info(f"Saved best overall DQN model to: {model_path}")

    # Chuyển numpy array
    np_rewards = np.array(all_results["rewards"])
    np_losses = np.array(all_results["losses"])
    
    mean_rewards = np.mean(np_rewards, axis=0)
    std_rewards = np.std(np_rewards, axis=0)
    mean_losses = np.mean(np_losses, axis=0)
    std_losses = np.std(np_losses, axis=0)

    try:
        os.makedirs(os.path.join(RUNS_DIR, "models"), exist_ok=True)
        
        summary_df = pd.DataFrame({
            'episode': range(len(mean_rewards)),
            'mean_reward': mean_rewards,
            'std_reward': std_rewards,
            'mean_loss': mean_losses,
            'std_loss': std_losses
        })
        summary_path = os.path.join(RUNS_DIR, "models", "dqn_summary_results.csv")
        summary_df.to_csv(summary_path, index=False)
        if not IS_CI: mlflow.log_artifact(summary_path)

        for i, seed in enumerate(seeds):
            seed_df = pd.DataFrame({
                'episode': range(len(all_results["rewards"][i])),
                'reward': all_results["rewards"][i],
                'loss': all_results["losses"][i],
                'epsilon': all_results["epsilons"][i]
            })
            seed_path = os.path.join(RUNS_DIR, "models", f"dqn_seed_{seed}_results.csv")
            seed_df.to_csv(seed_path, index=False)
            if not IS_CI: mlflow.log_artifact(seed_path)

        metrics_df = pd.DataFrame({
            'seed': seeds,
            'final_reward': [all_results["rewards"][i][-1] for i in range(len(seeds))],
            'best_reward': [max(all_results["rewards"][i]) for i in range(len(seeds))],
            'mean_reward': [np.mean(all_results["rewards"][i]) for i in range(len(seeds))],
            'final_loss': [all_results["losses"][i][-1] for i in range(len(seeds))],
            'mean_loss': [np.mean(all_results["losses"][i]) for i in range(len(seeds))]
        })
        metrics_path = os.path.join(RUNS_DIR, "models", "dqn_metrics_by_seed.csv")
        metrics_df.to_csv(metrics_path, index=False)
        if not IS_CI: mlflow.log_artifact(metrics_path)

    except Exception as e:
        logging.error(f"Failed to save CSV results: {e}")

    logging.info(f"\nĐã hoàn thành DQN Multi-seed training")
    logging.info(f"Final Mean Reward: {float(mean_rewards[-1]):.2f} ± {float(std_rewards[-1]):.2f}")
    
    # MLflow Model Registration
    if not IS_CI and best_agent_overall is not None:
        try:
            print("[*] Logging model to MLflow...")
            mlflow_pytorch_log_model(best_agent_overall.q_net, artifact_path="model")
            mlflow.log_metric("final_mean_reward", float(mean_rewards[-1]))
            mlflow.log_metric("best_mean_reward", float(np.max(mean_rewards)))

            if os.path.exists(model_path): mlflow.log_artifact(model_path)
            if os.path.exists(data_path): mlflow.log_artifact(data_path)

            current_run = mlflow.active_run()
            if current_run is None:
                raise RuntimeError("Không tìm thấy Active Run!")
            
            run_id = current_run.info.run_id

            try:
                client.get_registered_model("SDN_DQN_Model")
            except Exception:
                client.create_registered_model("SDN_DQN_Model")

            client.create_model_version(
                name="SDN_DQN_Model",
                source=f"{mlflow.get_artifact_uri()}/model",
                run_id=run_id
            )
            print("[+] Model Version Registered Successfully!")

        except Exception as e:
            print("[-] MLflow logging failed. Traceback:")
            traceback.print_exc()

    if not IS_CI:
        mlflow.end_run()

def promote_best_model():
    best_model, _ = get_best_model()
    
    # FIX: Check None để loại bỏ warning từ Pylance và tránh lỗi runtime
    if best_model is None:
        print("[-] Không tìm thấy model nào để promote lên Production.")
        return
        
    client.transition_model_version_stage(
        name=best_model.name,
        version=best_model.version,
        stage="Production"
    )
    print(f"Promote {best_model.name} v{best_model.version} → Production")

def rollback_model(model_name):
    versions = client.search_model_versions(f"name='{model_name}'")
    
    # FIX: Đảm bảo versions không bị rỗng hoặc None
    if not versions or len(versions) < 2:
        print("Không đủ version để rollback")
        return

    previous = sorted(versions, key=lambda x: int(x.version))[-2]
    client.transition_model_version_stage(
        name=model_name,
        version=previous.version,
        stage="Production"
    )
    print(f"Rollback → {model_name} v{previous.version}")    

if __name__ == "__main__":
    PORT = 9002
    start_http_server(PORT)
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"[DQN] Prometheus metrics server started on port {PORT}")
    train_multi_seeds_dqn()