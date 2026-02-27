import mlflow
from agent import DQNAgent
from env import SDNEnv

agent = DQNAgent()
env = SDNEnv()

mlflow.start_run()

for episode in range(5):

    state = env.reset()
    total_reward = 0

    for t in range(5):

        action = agent.select_action(state)

        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward

    print(f"Episode {episode} reward:", total_reward)
    mlflow.log_metric("episode_reward", total_reward, step=episode)

mlflow.end_run()