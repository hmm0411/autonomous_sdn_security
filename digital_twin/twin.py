import os

import joblib
import numpy as np


class DigitalTwin:
    """
    Digital Twin surrogate model.

    Input:
      X = [state_8, action]
      shape = 9 features

    Output:
      [next_latency, next_packet_loss]
    """

    def __init__(self, model_path="models/surrogate_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Surrogate model not found: {model_path}. "
                f"Run: python3 -m digital_twin.train_surrogate first."
            )

        self.model_path = model_path
        self.model = joblib.load(model_path)

        print(f"[TWIN] Loaded surrogate model: {model_path}", flush=True)

    def simulate(self, state, action):
        state_arr = np.asarray(state, dtype=float).reshape(-1)

        if len(state_arr) != 8:
            raise ValueError(
                f"DigitalTwin expects 8-dim state, got {len(state_arr)}"
            )

        x = np.concatenate(
            [
                state_arr,
                np.array([float(action)], dtype=float),
            ],
            axis=0,
        ).reshape(1, -1)

        prediction = self.model.predict(x)[0]

        return {
            "latency": float(prediction[0]),
            "packet_loss": float(prediction[1]),
        }