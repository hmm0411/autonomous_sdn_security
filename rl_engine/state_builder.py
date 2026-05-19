from collections import deque
import os
import joblib
import numpy as np


class StateBuilder:
    def __init__(self, scaler_path="models/scaler.pkl"):
        self.prev_action = 0

        self.packet_buf = deque(maxlen=3)
        self.byte_buf = deque(maxlen=3)
        self.cpu_buf = deque(maxlen=3)
        self.flow_growth_buf = deque(maxlen=5)

        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"[+] Loaded scaler: {scaler_path}")
        else:
            self.scaler = None
            print(f"[WARN] Scaler not found: {scaler_path}")

    def _preprocess_raw(self, raw):
        packet_rate = np.log1p(max(0.0, float(raw.get("packet_rate", 0.0))))
        byte_rate = np.log1p(max(0.0, float(raw.get("byte_rate", 0.0))))

        flow_count = np.clip(float(raw.get("flow_count", 0.0)), 0, 500)
        flow_growth_rate = np.clip(float(raw.get("flow_growth_rate", 0.0)), 0, 50)
        src_ip_entropy = np.clip(float(raw.get("src_ip_entropy", 0.0)), 0, 8)
        latency = np.clip(float(raw.get("latency", 0.0)), 0, 100)
        packet_loss = np.clip(float(raw.get("packet_loss", 0.0)), 0, 0.5)
        controller_cpu = np.clip(float(raw.get("controller_cpu", 0.0)), 0, 100)

        self.packet_buf.append(packet_rate)
        self.byte_buf.append(byte_rate)
        self.cpu_buf.append(controller_cpu)
        self.flow_growth_buf.append(flow_growth_rate)

        packet_rate = float(np.mean(self.packet_buf))
        byte_rate = float(np.mean(self.byte_buf))
        controller_cpu = float(np.mean(self.cpu_buf))
        flow_growth_rate = float(np.max(self.flow_growth_buf))

        return np.array([[
            packet_rate,
            byte_rate,
            flow_count,
            flow_growth_rate,
            src_ip_entropy,
            latency,
            packet_loss,
            controller_cpu,
        ]], dtype=np.float32)

    def build(self, raw):
        raw_vector = self._preprocess_raw(raw)

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

    def update_action(self, action):
        self.prev_action = action