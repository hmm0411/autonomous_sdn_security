import time
import numpy as np
import datetime
from rl_client import get_action
from metrics import update_metrics
from prometheus_client import start_http_server, Gauge
# Import từ thư mục rl_engine của bạn
from rl_engine.state_builder import StateBuilder
from rl_engine.controller_client import ControllerClient
from rl_engine.reward import Reward
from rl_engine.logger import Logger

from llm.prompt_builder import build_prompt
from llm.llm_service import call_llm

def main():
    print("Bắt đầu chạy Control Loop với LLM Cognition...")
    
    # Khởi tạo các class từ rl_engine
    state_builder = StateBuilder()
    controller = ControllerClient()
    reward_calc = Reward()
    logger = Logger(log_dir="results/logs") # Lưu log cho mục tiêu báo cáo

    while True:
        try:
            # 1. Build state
            raw_data = controller.get_state()
            state = state_builder.build(raw_data)
            
            # 2. Gọi RL Serving API (đã container hóa)
            action = get_action(state)
            
            # 3. Tích hợp LLM
            if action != 0: 
                # Trích xuất chỉ số QoS từ raw_data để LLM đánh giá
                qos_metrics = {
                    "delay": raw_data.get("latency", 0),
                    "loss": raw_data.get("packet_loss", 0),
                    "throughput": raw_data.get("byte_rate", 0)
                }
                
                # Tạo prompt và gọi LLM giải thích
                prompt = build_prompt(state, action, qos_metrics)
                explanation = call_llm(prompt)
                
                print(f"\nLLM SECURITY REPORT")
                print(explanation)

                # Lưu file log
                try:
                    with open("logs/llm_reports.log", "a", encoding="utf-8") as f:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{timestamp}] LLM SECURITY REPORT\n")
                        f.write(str(explanation).strip() + "\n")
                        f.write("-" * 50 + "\n")
                        f.flush()
                except Exception as file_err:
                    print(f"Lỗi khi ghi file log: {file_err}")
                
                # Lưu vào log 
                logger.log_llm(episode=0, step=0, state=state, action=action, 
                               qos=qos_metrics, explanation=explanation)

            # 4. Apply xuống ONOS
            controller.apply_action(action)
            
            # 5. Tính toán Reward & xuất Metrics
            reward = reward_calc.calculate(raw_data, action)
            update_metrics(state, reward)
            
            print(f"\nState: {state} | Action: {action} | Reward: {reward}")
            
        except Exception as e:
            print(f"Lỗi vòng lặp: {e}")
            
        time.sleep(5)

if __name__ == "__main__":
    start_http_server(8000)
    main()
