import os

import numpy as np
import requests


STRICT_RL = os.getenv("STRICT_RL", "false").lower() == "true"

RL_ENDPOINTS = {
    "dqn": os.getenv("DQN_ENDPOINT", "http://rl-serving-dqn:8000/predict"),
    "ppo": os.getenv("PPO_ENDPOINT", "http://rl-serving-ppo:8001/predict"),
}


def get_action(state, model_type="dqn"):
    model_type = str(model_type).lower()
    url = RL_ENDPOINTS.get(model_type, RL_ENDPOINTS["dqn"])

    try:
        payload = {
            "state": state.tolist()
            if isinstance(state, np.ndarray)
            else list(state)
        }

        response = requests.post(
            url,
            json=payload,
            timeout=float(os.getenv("RL_TIMEOUT", "2")),
        )

        if response.status_code == 200:
            data = response.json()

            return (
                int(data.get("action", 0)),
                int(data.get("action_staging", 0)),
                str(data.get("model", model_type)).lower(),
            )

        print(
            f"[RL_API_ERROR] status={response.status_code} body={response.text}",
            flush=True,
        )

    except Exception as e:
        print(f"[RL_API_CONN_ERROR] model={model_type} error={e}", flush=True)

    if STRICT_RL:
        raise RuntimeError(f"Failed to get action from RL model: {model_type}")

    return 0, 0, f"{model_type}_fallback"