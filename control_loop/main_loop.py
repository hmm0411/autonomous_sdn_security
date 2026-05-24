import time
import numpy as np
import requests
from rl_engine.state_builder import StateBuilder
from control_loop.controller_client import execute_action
from rl_engine.reward import Reward

from control_loop.rl_client import get_action
from control_loop.metrics import update_metrics
from control_loop.state_collector import get_state

STATE_DIM = 8
SLEEP_TIME = 2

state_builder = StateBuilder()
reward_calc = Reward()

print("AUTO MODEL CONTROL LOOP STARTED")

def baseline_policy(state):
    return 0

def validate_state(state):
    return state is not None and len(state) == STATE_DIM

print("Đang chờ ONOS khởi động (khoảng 30 giây)...")
while True:
    try:
        res = requests.get("http://controller:8181/onos/v1/flows", auth=("onos", "rocks"), timeout=2)
        if res.status_code in [200, 401]:
            print("ONOS đã sẵn sàng!")
            break
    except Exception:
        print("ONOS chưa lên, đợi thêm 5 giây...")
        time.sleep(5)

# --- THÊM ĐOẠN NÀY ĐỂ ĐỢI API SERVING SẴN SÀNG ---
print("Đang chờ RL API Serving (DQN) khởi động...")
while True:
    try:
        # Gọi thử vào endpoint /health
        res = requests.get("http://rl-serving-dqn:8000/health", timeout=2)
        if res.status_code == 200:
            print("RL API Serving (DQN) đã sẵn sàng!")
            break
    except Exception:
        print("RL API chưa sẵn sàng, đợi 5 giây...")
        time.sleep(5)
# --------------------------------------------------

while True:
    raw = get_state()
    state = np.array(state_builder.build(raw), dtype=np.float32)

    model_to_use = "dqn"

    # Lấy action từ API
    action_prod, action_staging, model_name = get_action(state, model_type=model_to_use)

    # Tính toán phần thưởng
    reward_prod = reward_calc.calculate(raw, action_prod)
    reward_staging = reward_calc.calculate(raw, action_staging)

    execute_action(action_prod)
    update_metrics(state, reward_prod, reward_staging, model_name, action_prod)

    print(f"[{model_name}] Prod Action={action_prod} (R={reward_prod}) | Staging Action={action_staging} (R={reward_staging})")
    time.sleep(SLEEP_TIME)