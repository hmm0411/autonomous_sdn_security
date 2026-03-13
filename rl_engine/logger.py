import csv


class Logger:

    def __init__(self):

        self.ppo_logs = []
        self.dqn_logs = []

    # PPO logging
    def log_ppo(self, episode, reward, policy_loss, value_loss, entropy, actions):

        log = {
            "episode": episode,
            "reward": reward,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actions": actions
        }

        self.ppo_logs.append(log)

    # DQN logging
    def log_dqn(self, episode, reward, loss, epsilon, actions):

        log = {
            "episode": episode,
            "reward": reward,
            "loss": loss,
            "epsilon": epsilon,
            "actions": actions
        }

        self.dqn_logs.append(log)

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