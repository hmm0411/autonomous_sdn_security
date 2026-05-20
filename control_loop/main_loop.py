import time
import numpy as np

from rl_engine.state_builder import StateBuilder
from control_loop.controller_client import execute_action
from rl_engine.reward import Reward

from control_loop.rl_client import call_model
from control_loop.metrics import update_metrics
from control_loop.state_collector import get_state

STATE_DIM = 9
SLEEP_TIME = 2

RL_URL = "http://localhost:8000/predict"

state_builder = StateBuilder()
reward_calc = Reward()

print("AUTO MODEL CONTROL LOOP STARTED")

def baseline_policy(state):
    return 0
    
def validate_state(state):
    return state is not None and len(state) == STATE_DIM

while True:
    raw = get_state()
    state = np.array(state_builder.build(raw), dtype=np.float32)

    if not validate_state(state):
        continue

    action = call_model(RL_URL, state)

    reward = reward_calc.calculate(raw, action)

    execute_action(action)

    update_metrics(state, reward, "AUTO", action)

    print(f"[AUTO] action={action} | reward={reward}")

    time.sleep(SLEEP_TIME)