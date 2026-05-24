import requests
import numpy as np

RL_ENDPOINTS = {
    "dqn": "http://rl-agent-dqn:8000/predict",
    "ppo": "http://rl-agent-ppo:8001/predict"
}

def get_action(state, model_type="dqn"):
    # Cần xác định port dựa trên loại model
    port = 8000 if model_type == "dqn" else 8001
    # Hostname cũng nên linh hoạt theo service:
    host = "rl-serving-dqn" if model_type == "dqn" else "rl-serving-ppo"
    url = f"http://{host}:{port}/predict"
    
    try:
        payload = {"state": state.tolist() if isinstance(state, np.ndarray) else state}
        res = requests.post(url, json=payload, timeout=2.0)
        
        if res.status_code == 200:
            data = res.json()
            return int(data.get("action", 0)), int(data.get("action_staging", 0)), data.get("model", model_type)
    except Exception as e:
        print(f"RL API Connection error for {model_type}: {e}")
    
    return 0, 0, model_type