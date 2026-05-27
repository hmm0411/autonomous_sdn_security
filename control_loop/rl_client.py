import requests
import numpy as np

RL_ENDPOINTS = {
    "dqn": "http://rl-serving-dqn:8000/predict",
    "ppo": "http://rl-serving-ppo:8001/predict",
}

def get_action(state, model_type="dqn"):
    model_type = str(model_type).lower()
    url = RL_ENDPOINTS.get(model_type, RL_ENDPOINTS["dqn"])

    try:
        payload = {
            "state": state.tolist() if isinstance(state, np.ndarray) else state
        }

        res = requests.post(url, json=payload, timeout=2.0)

        if res.status_code == 200:
            data = res.json()
            return (
                int(data.get("action", 0)),
                int(data.get("action_staging", 0)),
                str(data.get("model", model_type)).lower()
            )

        print(f"RL API returned status={res.status_code}, body={res.text}")

    except Exception as e:
        print(f"RL API connection error for {model_type}: {e}")

    return 0, 0, model_type