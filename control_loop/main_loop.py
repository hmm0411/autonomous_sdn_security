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
raw = get_state()  # Lấy state đầu tiên để khởi tạo
state = state_builder.build(raw)

print("AUTO MODEL CONTROL LOOP STARTED")

def validate_state(state):
    return state is not None and len(state) == STATE_DIM

while True:
    try:
        raw = get_state()  # Lấy state mới
        state = np.array(state_builder.build(raw), dtype=np.float32)

        if not validate_state(state):
            print("Invalid state:", state)
            time.sleep(SLEEP_TIME)
            continue

        # AUTO SELECT MODEL
        action, model, reward = get_best_action(
            state,
            reward_calc.calculate
        )

        # APPLY
        execute_action(action)

        # METRICS
        update_metrics(state, reward, model)

        print(f"[AUTO] {model} | action={action} | reward={reward}")

        time.sleep(SLEEP_TIME)

    except Exception as e:
        print("Loop error:", e)
        time.sleep(3)