# init_model.py
import mlflow
import torch
import os
from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.config import STATE_DIM, ACTION_DIM
from dotenv import load_dotenv

load_dotenv()  # Load biến môi trường từ .env
mlflow.set_tracking_uri("http://35.240.135.171:30007") # Dùng port NodePort
mlflow.set_experiment("SDN_Autonomous_Security")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://35.240.135.171:30005"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

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