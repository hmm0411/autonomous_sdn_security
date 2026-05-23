import requests
import numpy as np

RL_ENDPOINTS = {
    "dqn": "http://rl-agent-dqn:8000/predict",
    "ppo": "http://rl-agent-ppo:8000/predict"
}

def get_action(state, model_type="dqn"):
    url = RL_ENDPOINTS.get(model_type, RL_ENDPOINTS["dqn"])
    try:
        payload = {"state": state.tolist() if isinstance(state, np.ndarray) else state}
        res = requests.post(url, json=payload, timeout=2.0)
        
        if res.status_code == 200:
            data = res.json()
            return int(data.get("action", 0)), data.get("model", model_type)
    except Exception as e:
        print(f"RL API Connection error for {model_type}: {e}")
    
    return 0, "fallback"