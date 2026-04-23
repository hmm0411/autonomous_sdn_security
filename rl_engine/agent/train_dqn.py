import os
import random
import logging
import numpy as np
import pandas as pd
import torch
import mlflow
from prometheus_client import start_http_server, Gauge
import collections

from rl_engine.env import SDNEnv
from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.offline_env import OfflineSDNEnv
from rl_engine.replay_buffer import ReplayBuffer
from rl_engine.logger import Logger
from rl_engine.config import *
from rl_engine.utils import set_seed

PROM_REWARD = Gauge('episode_reward', 'Phần thưởng của Episode', ['agent'])
PROM_LOSS = Gauge('training_loss', 'Loss của mô hình', ['agent'])
PROM_STEPS = Gauge('episode_steps', 'Số bước sống sót trong episode', ['agent'])

logging.basicConfig(level=logging.INFO)

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("sdn-rl-dqn")

def run_single_seed_dqn(seed_value, df_train, parent_run=None):
    """Huấn luyện DQN với một Seed cụ thể"""
    set_seed(seed_value)
    env = OfflineSDNEnv(dataframe=df_train)
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    logger = Logger(log_dir=f"../../runs/dqn_seed_{seed_value}")
    
    epsilon = EPS_START
    seed_reward_history = []
    seed_loss_history = []
    seed_epsilon_history = []
    best_reward = -float('inf')
    recent_rewards = collections.deque(maxlen=10)
    
    total_episodes = 2 if os.getenv("CI") == "true" else MAX_EPISODES

    for episode in range(total_episodes):
        state, _ = env.reset(seed=seed_value if episode == 0 else None)
        total_reward = 0
        losses = []
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

            state = next_state
            total_reward += reward

            if len(buffer) > BATCH_SIZE:
                loss = agent.update(buffer.sample(BATCH_SIZE))
                losses.append(loss)

            if done:
                break

        # Cập nhật epsilon
        epsilon = max(EPS_END, epsilon - (EPS_START - EPS_END) / EPS_DECAY)
        seed_reward_history.append(total_reward)
        recent_rewards.append(total_reward)
        
        # Calculate average loss for episode
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        seed_loss_history.append(avg_loss)
        seed_epsilon_history.append(epsilon)

        logger.log_dqn(
            episode=episode,
            reward=total_reward,
            loss=avg_loss,
            epsilon=epsilon,
            actions=actions_in_episode
        )

        # Log metrics to MLflow if in parent run context
        if parent_run and mlflow.active_run() is not None:
            try:
                mlflow.log_metric(f"reward_seed_{seed_value}", total_reward, step=episode)
                mlflow.log_metric(f"loss_seed_{seed_value}", avg_loss, step=episode)
                mlflow.log_metric(f"epsilon_seed_{seed_value}", epsilon, step=episode)
                mlflow.log_metric(f"buffer_size_seed_{seed_value}", len(buffer), step=episode)
            except Exception as e:
                logging.warning(f"Failed to log metrics to MLflow: {e}")

        # Save best model checkpoint
        if len(recent_rewards) == 10:
            avg_reward = np.mean(recent_rewards)
            if avg_reward > best_reward and episode > 20:
                best_reward = avg_reward
                save_path = f"../../models/best_dqn_seed_{seed_value}.pth"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                try:
                    agent.save(save_path)
                    logging.info(f"[Seed {seed_value}] Ep {episode}: New Best Avg Reward: {best_reward:.2f}")
                except Exception as e:
                    logging.error(f"Failed to save best model: {e}")

        if episode % 50 == 0:
            logging.info(
                f"Seed {seed_value} | Ep {episode} | Reward: {total_reward:.2f} | "
                f"Eps: {epsilon:.3f} | Loss: {avg_loss:.4f}"
            )

    logger.close()
    return {
        "rewards": seed_reward_history,
        "losses": seed_loss_history,
        "epsilons": seed_epsilon_history
    }

def train_multi_seeds_dqn():
    """Train DQN with multiple seeds and log all metrics"""
    seeds = [42, 101, 123, 456, 789]
    all_results = {
        "rewards": [],
        "losses": [],
        "epsilons": []
    }
    df_train = pd.read_csv("../../data/processed/train_data.csv")
    
    os.makedirs("../../results", exist_ok=True)
    
    with mlflow.start_run(run_name="DQN_MultiSeed_Analysis"):
        mlflow.log_param("algo", "DQN")
	mlflow.log_param("algo", "DQN") # hoặc "PPO"
	mlflow.log_param("state_dim", STATE_DIM)
	mlflow.log_param("action_dim", ACTION_DIM)
        
	mlflow.log_param("num_seeds", len(seeds))
        mlflow.log_param("seeds", str(seeds))
        mlflow.log_param("buffer_size", BUFFER_SIZE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("eps_start", EPS_START)
        mlflow.log_param("eps_end", EPS_END)
        mlflow.log_param("eps_decay", EPS_DECAY)
        
        mlflow.pytorch.log_model(model, "dqn_model")
       
        run_id = mflow.active_run().info.run_id
	model_uri = f"runs:/{run_id}/dqn_model"
	mlflow.register_model(model_uri, "SDN_DQN_Model")

        for s in seeds:
            logging.info(f"\n--- BẮT ĐẦU DQN SEED: {s} ---")
            history = run_single_seed_dqn(s, df_train, parent_run=True)
            all_results["rewards"].append(history["rewards"])
            all_results["losses"].append(history["losses"])
            all_results["epsilons"].append(history["epsilons"])
            
            # Log final metrics for each seed
            mlflow.log_metric(f"final_reward_seed_{s}", history["rewards"][-1])
            mlflow.log_metric(f"final_loss_seed_{s}", history["losses"][-1])
            mlflow.log_metric(f"final_epsilon_seed_{s}", history["epsilons"][-1])
            mlflow.log_metric(f"best_reward_seed_{s}", max(history["rewards"]))

        # Calculate statistics across all seeds
        all_rewards = np.array(all_results["rewards"])
        all_losses = np.array(all_results["losses"])
        all_epsilons = np.array(all_results["epsilons"])
        
        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)
        mean_losses = np.mean(all_losses, axis=0)
        std_losses = np.std(all_losses, axis=0)
        
        # Log aggregated metrics
        mlflow.log_metric("final_mean_reward", mean_rewards[-1])
        mlflow.log_metric("final_std_reward", std_rewards[-1])
        mlflow.log_metric("best_mean_reward", np.max(mean_rewards))
        mlflow.log_metric("best_std_reward", std_rewards[np.argmax(mean_rewards)])

        # Save comprehensive results to CSV files
        try:
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
            mlflow.log_artifact(summary_path)
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
                mlflow.log_artifact(seed_path)
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
            mlflow.log_artifact(metrics_path)
            logging.info(f"Saved detailed metrics to: {metrics_path}")

        except Exception as e:
            logging.error(f"Failed to save results: {e}")

        logging.info(f"\n✓ Đã hoàn thành DQN Multi-seed training")
        logging.info(f"Final Mean Reward: {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")
        logging.info(f"Best Mean Reward: {np.max(mean_rewards):.2f}")

if __name__ == "__main__":
    start_http_server(9000)
    logging.info(f"Đã khởi động Prometheus metrics server trên port {port}")
    train_multi_seeds_dqn()
