import time
import numpy as np

from control_loop.controller_client import execute_action
from control_loop.rl_client import get_best_action
from control_loop.state_collector import get_state
from control_loop.metrics import update_metrics
from rl_engine.reward import Reward

STATE_DIM = 9
SLEEP_TIME = 2

reward_calc = Reward()
warmup_counter = 0


def norm(x, max_val):
    return min(max(x / max_val, 0.0), 1.0)


while True:
    try:
        raw = get_state()

        if raw is None:
            time.sleep(SLEEP_TIME)
            continue

        # ===== ONLINE NORMALIZATION =====
        state = np.array([
            norm(raw["packet_rate"], 20000),
            norm(raw["byte_rate"], 500000),
            norm(raw["flow_count"], 100),
            norm(raw["latency"], 100),
            np.clip(raw["packet_loss"], 0, 1),
            np.clip(raw["src_ip_entropy"], 0, 1),
            np.clip(raw["queue_length"], 0, 1),
            np.clip(raw["controller_cpu"], 0, 1),
            0  # previous_action (optional)
        ], dtype=np.float32)

        print("STATE:", state)

        warmup_counter += 1
        if warmup_counter < 5:
            print("Warmup...")
            time.sleep(SLEEP_TIME)
            continue

        action, model_name, _ = get_best_action(state)

        execute_action(action)

        time.sleep(SLEEP_TIME)

        next_raw = get_state()
        reward = reward_calc.calculate(raw, action, next_raw)

        print(f"[AUTO] {model_name} | action={action} | reward={reward}")

        update_metrics(state, reward, model_name, action)

    except Exception as e:
        print("Loop error:", e)
        time.sleep(3)