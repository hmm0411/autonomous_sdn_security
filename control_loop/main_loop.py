import time
import numpy as np
from rl_client import get_action
from metrics import update_metrics
from prometheus_client import start_http_server, Gauge
# Import từ thư mục rl_engine của bạn
from rl_engine.state_builder import StateBuilder
from rl_engine.controller_client import ControllerClient
from rl_engine.reward import Reward

def main():
    print("Bắt đầu chạy Control Loop...")
    
    # Khởi tạo các class từ rl_engine
    state_builder = StateBuilder()
    controller = ControllerClient()
    reward_calc = Reward()

    while True:
        try:
            # 1. Build state
            state = state_builder.build()
            
            # 2. Gọi RL Serving API (đã container hóa)
            action = get_action(state)
            
            # 3. Apply xuống ONOS
            controller.apply_action(action)
            
            # 4. Tính toán Reward & xuất Metrics
            reward = reward_calc.compute(state, action)
            update_metrics(state, reward)
            
            print(f"State: {state} | Action: {action} | Reward: {reward}")
            
        except Exception as e:
            print(f"Lỗi vòng lặp: {e}")
            
        time.sleep(2)

if __name__ == "__main__":
    start_http_server(8000)
    main()
