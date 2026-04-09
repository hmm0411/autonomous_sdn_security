import csv
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger:

    def __init__(self, log_dir="runs/experiment"):
        """Khởi tạo TensorBoard Writer"""
        # Đảm bảo thư mục runs/ tồn tại
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.llm_logs = []

    # PPO logging
    def log_ppo(self, episode, reward, policy_loss, value_loss, entropy, actions):

        self.writer.add_scalar("Train/Total_Reward", reward, episode)
        self.writer.add_scalar("Loss/Policy_Loss", policy_loss, episode)
        self.writer.add_scalar("Loss/Value_Loss", value_loss, episode)
        self.writer.add_scalar("Metrics/Entropy", entropy, episode)
        self.writer.add_histogram("Actions", np.array(actions), episode)

    # DQN logging
    def log_dqn(self, episode, reward, loss, epsilon, actions):

        self.writer.add_scalar("Train/Total_Reward", reward, episode)
        self.writer.add_scalar("Loss/MSE_Loss", loss, episode)
        self.writer.add_scalar("Metrics/Epsilon", epsilon, episode)
        self.writer.add_histogram("Actions", np.array(actions), episode)

    # LLM logging
    def log_llm(self, episode, step, state, action, qos, explanation):

        # parse SAFE / RISKY
        safety = -1
        if "SAFE" in explanation.upper():
            safety = 1
        elif "RISKY" in explanation.upper():
            safety = 0

        # TensorBoard
        self.writer.add_scalar("LLM/Safety", safety, episode)

        self.llm_logs.append({
            "episode": episode,
            "step": step,
            "state": state.tolist() if hasattr(state, "tolist") else state,
            "action": int(action),
            "qos": qos,
            "explanation": explanation,
            "safety": safety
        })

    # Save PPO logs
    def save_ppo(self, filename="ppo_training_log.csv"):

        with open(filename, "w", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "reward",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "actions"
                ]
            )

            writer.writeheader()

            for row in self.ppo_logs:
                writer.writerow(row)

    # Save DQN logs
    def save_dqn(self, filename="dqn_training_log.csv"):

        with open(filename, "w", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "reward",
                    "loss",
                    "epsilon",
                    "actions"
                ]
            )

            writer.writeheader()

            for row in self.dqn_logs:
                writer.writerow(row)

    def close(self):
        """Đóng writer khi kết thúc huấn luyện"""
        self.writer.close()

    # Save LLM
    def save_llm(self, filename="llm_log.csv"):

        with open(filename, "w", newline="", encoding="utf-8") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "step",
                    "state",
                    "action",
                    "qos",
                    "explanation",
                    "safety"
                ]
            )

            writer.writeheader()
            for row in self.llm_logs:
                writer.writerow(row)

    def close(self):
        self.writer.close()