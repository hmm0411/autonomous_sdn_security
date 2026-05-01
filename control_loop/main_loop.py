import time
import os
import datetime
import numpy as np

from rl_engine.state_builder import StateBuilder
from control_loop.controller_client import execute_action
from rl_engine.reward import Reward
from rl_engine.logger import Logger

from llm.prompt_builder import build_prompt
from llm.llm_service import call_llm

from control_loop.rl_client import get_best_action
from control_loop.metrics import update_metrics
from control_loop.state_collector import get_state

STATE_DIM = 9
SLEEP_TIME = 2

state_builder = StateBuilder()
reward_calc = Reward()
logger = Logger(log_dir="results/logs")

print("AUTO MODEL CONTROL LOOP STARTED")

def baseline_policy(state):
    return 0

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

        # ===== TÍCH HỢP LLM =====
        if action != 0:
            qos_metrics = {
                "delay": raw.get("latency", 0),
                "loss": raw.get("packet_loss", 0),
                "throughput": raw.get("byte_rate", 0)
            }

            prompt = build_prompt(state, action, qos_metrics)
            explanation = call_llm(prompt)

            print(f"\nLLM SECURITY REPORT")
            print(explanation)

            try:
                os.makedirs("logs", exist_ok=True)
                with open("logs/llm_reports.log", "a", encoding="utf-8") as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] LLM SECURITY REPORT\n")
                    f.write(str(explanation).strip() + "\n")
                    f.write("-" * 50 + "\n")
                    f.flush()
            except Exception as file_err:
                print(f"Lỗi khi ghi file log: {file_err}")

            logger.log_llm(episode=0, step=0, state=state, action=action,
                           qos=qos_metrics, explanation=explanation)

        # ===== BASELINE =====
        action_base = baseline_policy(state)
        reward_base = reward_calc.calculate(raw, action_base)

        # ===== APPLY =====
        execute_action(action)

        # ===== UPDATE METRICS =====
        update_metrics(state, reward, model, action)
        update_metrics(state, reward_base, "baseline", action_base)

        print(f"[AUTO] {model} | action={action} | reward={reward}")
        print(f"[BASELINE] action={action_base} | reward={reward_base}")

        time.sleep(SLEEP_TIME)

    except Exception as e:
        print("Loop error:", e)
        time.sleep(3)