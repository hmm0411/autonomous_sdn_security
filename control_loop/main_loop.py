import time
import os
import datetime
import joblib
import numpy as np

from control_loop.controller_client import execute_action
from rl_engine.reward import Reward
from rl_engine.logger import Logger
from control_loop.rl_client import get_best_action
from control_loop.state_collector import get_state
from control_loop.metrics import update_metrics

STATE_DIM = 9
SLEEP_TIME = 2

DQN_URL = "http://rl-agent-dqn:9000/predict"
PPO_URL = "http://rl-agent-ppo:9001/predict"

# ===== LOAD SCALER =====
scaler = joblib.load("models/scaler.pkl")

reward_calc = Reward()
logger = Logger(log_dir="results/logs")

print("AUTO MODEL CONTROL LOOP STARTED")


def validate_state(state):
    return state is not None and len(state) == STATE_DIM


def scale_state(raw_dict):
    raw_vector = np.array([[
        raw_dict["packet_rate"],
        raw_dict["byte_rate"],
        raw_dict["flow_count"],
        raw_dict["src_ip_entropy"],
        raw_dict["latency"],
        raw_dict["packet_loss"],
        raw_dict["queue_length"],
        raw_dict["controller_cpu"],
    ]], dtype=np.float32)

    scaled_core = scaler.transform(raw_vector)[0]
    previous_action = raw_dict.get("previous_action", 0)

    return np.append(scaled_core, previous_action)


while True:
    try:
        # =============================
        # GET CURRENT STATE s_t
        # =============================
        raw_state = get_state()

        if raw_state is None:
            print("State is None, skipping...")
            time.sleep(SLEEP_TIME)
            continue

        state = scale_state(raw_state)

        if not validate_state(state):
            print("Invalid state:", state)
            time.sleep(SLEEP_TIME)
            continue

        print("STATE (scaled):", state)

        # =============================
        # SELECT ACTION (PPO ưu tiên)
        # =============================
        action, model_name, _ = get_best_action(state)

        # =============================
        # HYBRID DECISION
        # =============================

        flow_ratio = state[2]
        entropy = state[3]
        queue_ratio = state[6]
        cpu = state[7]
        # =============================
        # EXECUTE ACTION
        # =============================
        execute_action(action)

        # =============================
        # WAIT ENV RESPONSE
        # =============================
        time.sleep(SLEEP_TIME)

        # =============================
        # GET NEXT STATE s_{t+1}
        # =============================
        next_raw_state = get_state()

        if next_raw_state is None:
            print("Next state is None")
            continue

        # =============================
        # COMPUTE REAL REWARD
        # =============================
        reward = reward_calc.calculate(
            prev_state=raw_state,
            action=action,
            next_state=next_raw_state
        )

        print(f"[AUTO] {model_name} | action={action} | reward={reward}")

        # =============================
        # LOG
        # =============================
        update_metrics(state, reward, model_name, action)

    except Exception as e:
        print("Loop error:", e)
        time.sleep(3)