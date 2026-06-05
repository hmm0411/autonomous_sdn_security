import os
from typing import Any, Tuple

import numpy as np
import requests


RL_ENDPOINTS = {
    "dqn": os.getenv("DQN_ENDPOINT", "http://rl-serving-dqn:8000/predict"),
    "ppo": os.getenv("PPO_ENDPOINT", "http://rl-serving-ppo:8001/predict"),
}


def _state_to_list(state: Any) -> list:
    if isinstance(state, np.ndarray):
        return state.astype(float).reshape(-1).tolist()

    if isinstance(state, list):
        return state

    if isinstance(state, tuple):
        return list(state)

    return list(state)


def get_action(state, model_type="dqn") -> Tuple[int, int, str]:
    model_type = str(model_type).lower()
    url = RL_ENDPOINTS.get(model_type, RL_ENDPOINTS["dqn"])

    try:
        payload = {
            "state": _state_to_list(state)
        }

        print(f"[RL_REQUEST] model={model_type} url={url} state={payload['state']}", flush=True)

        response = requests.post(
            url,
            json=payload,
            timeout=float(os.getenv("RL_TIMEOUT", "3")),
        )

        if response.status_code == 200:
            data = response.json()

            action = int(data.get("action", 0))
            action_staging = int(data.get("action_staging", action))
            model = str(data.get("model", model_type)).lower()

            print(
                f"[RL_RESPONSE] model={model} action={action} staging={action_staging}",
                flush=True,
            )

            return action, action_staging, model

        print(
            f"[RL_API_ERROR] status={response.status_code} body={response.text[:300]}",
            flush=True,
        )

    except Exception as e:
        print(f"[RL_API_CONN_ERROR] model={model_type} url={url} error={e}", flush=True)

    return 0, 0, model_type