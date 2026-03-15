import numpy as np

from rl_engine.env import SDNEnv
from rl_engine.ppo_agent import PPOAgent
from rl_engine.logger import Logger
from rl_engine.config import STATE_DIM, ACTION_DIM


MAX_EPISODES = 1000
MAX_STEPS = 200


def train():

    env = SDNEnv()

    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM
    )

    logger = Logger()

    for episode in range(MAX_EPISODES):

        state = env.reset()

        states = []
        actions = []
        rewards = []
        log_probs = []
        dones = []

        action_history = []

        total_reward = 0

        for step in range(MAX_STEPS):

            action, log_prob = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

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

    logger.save_ppo()


if __name__ == "__main__":
    train()