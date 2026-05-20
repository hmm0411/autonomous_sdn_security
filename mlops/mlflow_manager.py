import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

stages=["None", "Staging"]

def get_best_model():
    dqn = client.get_latest_versions("SDN_DQN_Model", stages=stages)
    ppo = client.get_latest_versions("SDN_PPO_Model", stages=stages)

    best_model = None
    best_score = -1e9

    for model in dqn + ppo:
        run = client.get_run(model.run_id)
        reward = run.data.metrics.get("final_mean_reward", -1e9)

        if reward > best_score:
            best_score = reward
            best_model = model

    return best_model, best_score