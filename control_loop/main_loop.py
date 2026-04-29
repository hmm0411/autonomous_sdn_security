import time
import numpy as np

from rl_engine.state_builder import StateBuilder
from control_loop.controller_client import execute_action
from rl_engine.reward import Reward

from control_loop.rl_client import get_best_action
from control_loop.metrics import update_metrics
from control_loop.state_collector import get_state

STATE_DIM = 9
SLEEP_TIME = 2

state_builder = StateBuilder()
reward_calc = Reward()

print("AUTO MODEL CONTROL LOOP STARTED")

def validate_state(state):
    return state is not None and len(state) == STATE_DIM

while True:
    try:
        # ===== LẤY RAW DATA =====
        raw = get_state()

        # ===== BUILD STATE =====
        state = np.array(state_builder.build(raw), dtype=np.float32)

        if not validate_state(state):
            print("Invalid state:", state)
            time.sleep(SLEEP_TIME)
            continue

        # ===== AUTO CHỌN MODEL =====
        action, model, reward = get_best_action(
            state,
            lambda s, a: reward_calc.calculate(raw, a)
        )

        # ===== APPLY =====
        execute_action(action)

        # ===== UPDATE METRICS =====
        update_metrics(state, reward, model)

        print(f"[AUTO] {model} | action={action} | reward={reward}")

        time.sleep(SLEEP_TIME)

    except Exception as e:
        print("Loop error:", e)
        time.sleep(3)