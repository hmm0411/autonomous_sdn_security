# init_model.py
import mlflow
import torch
import os
from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.config import STATE_DIM, ACTION_DIM

mlflow.set_tracking_uri("http://localhost:30007") # Dùng port NodePort
mlflow.set_experiment("SDN_Autonomous_Security")

def init():
    agent = DQNAgent(STATE_DIM, ACTION_DIM)
    # Lưu model giả lập
    mlflow.pytorch.log_model(agent.q_net, "model", registered_model_name="SDN_DQN_Model")
    # Lưu scaler giả lập
    os.makedirs("models", exist_ok=True)
    import joblib
    from sklearn.preprocessing import QuantileTransformer
    joblib.dump(QuantileTransformer(), "models/scaler.pkl")
    print("Đã tạo model và scaler mồi cho MLflow!")

init()