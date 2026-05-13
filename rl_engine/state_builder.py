import os
import joblib
import numpy as np


FEATURES = [
    "packet_rate",
    "byte_rate",
    "flow_count",
    "flow_growth_rate",
    "src_ip_entropy",
    "latency",
    "packet_loss",
    "controller_cpu"
]


class StateBuilder:
    def __init__(self, scaler_path="models/scaler.pkl"):
        self.prev_action = 0
        self.scaler = None

        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"[+] Loaded scaler: {scaler_path}")
        else:
            print(f"[WARN] Scaler not found: {scaler_path}. Using raw state.")

    def build(self, raw):
        raw_vector = np.array([[
            float(raw.get("packet_rate", 0.0)),
            float(raw.get("byte_rate", 0.0)),
            float(raw.get("flow_count", 0.0)),
            float(raw.get("flow_growth_rate", 0.0)),
            float(raw.get("src_ip_entropy", 0.0)),
            float(raw.get("latency", 0.0)),
            float(raw.get("packet_loss", 0.0)),
            float(raw.get("controller_cpu", 0.0)),
        ]], dtype=np.float32)

        if self.scaler is not None:
            scaled = self.scaler.transform(raw_vector)[0]
        else:
            scaled = raw_vector[0]

        state = np.array([
            scaled[0],
            scaled[1],
            scaled[2],
            scaled[3],
            scaled[4],
            scaled[5],
            scaled[6],
            scaled[7],
            float(self.prev_action) / 4.0
        ], dtype=np.float32)

        return state