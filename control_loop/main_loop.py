import time
import os
import datetime
import joblib
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

# ===== LOAD SCALER =====
scaler = joblib.load("models/scaler.pkl")

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
        # ====================================================
        # LẤY RAW STATE DICTIONARY
        # ====================================================
        raw_dict = get_state()

        if raw_dict is None:
            print("State is None, skipping...")
            time.sleep(SLEEP_TIME)
            continue

        # ====================================================
        # BUILD RAW VECTOR (ĐÚNG THỨ TỰ TRAINING)
        # ====================================================
        raw_vector = np.array([[
            raw_dict["packet_rate"],
            raw_dict["byte_rate"],
            raw_dict["flow_count"],
            raw_dict["src_ip_entropy"],
            raw_dict["latency"],
            raw_dict["packet_loss"],
            raw_dict["queue_length"],
            raw_dict["controller_cpu"],
            raw_dict.get("previous_action", 0)
        ]], dtype=np.float32)

        # ====================================================
        # SCALE STATE (QUAN TRỌNG NHẤT)
        # ====================================================
        state_scaled = scaler.transform(raw_vector)[0]

        # DEBUG SCALE
        print("RAW:", raw_vector)
        print("SCALED:", state_scaled)

        state = state_scaled

        if not validate_state(state):
            print("Invalid state:", state)
            time.sleep(SLEEP_TIME)
            continue

        # ====================================================
        # AUTO MODEL SELECTION
        # ====================================================
        action, model, reward = get_best_action(
            state,
            lambda s, a: reward_calc.calculate(raw_dict, a)
        )

        # ====================================================
        # LLM EXPLANATION (NẾU CÓ ACTION)
        # ====================================================
        if action != 0:
            qos_metrics = {
                "delay": raw_dict["latency"],
                "loss": raw_dict["packet_loss"],
                "throughput": raw_dict["byte_rate"]
            }

            prompt = build_prompt(state.tolist(), action, qos_metrics)
            explanation = call_llm(prompt)

            print("\nLLM SECURITY REPORT")
            print(explanation)

            os.makedirs("logs", exist_ok=True)
            with open("logs/llm_reports.log", "a", encoding="utf-8") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] LLM SECURITY REPORT\n")
                f.write(str(explanation).strip() + "\n")
                f.write("-" * 50 + "\n")

            logger.log_llm(
                episode=0,
                step=0,
                state=state,
                action=action,
                qos=qos_metrics,
                explanation=explanation
            )

        # ====================================================
        # BASELINE
        # ====================================================
        action_base = baseline_policy(state)
        reward_base = reward_calc.calculate(raw_dict, action_base)

        # ====================================================
        # EXECUTE ACTION
        # ====================================================
        execute_action(action)

        # ====================================================
        # 8 UPDATE METRICS
        # ====================================================
        update_metrics(state, reward, model, action)
        update_metrics(state, reward_base, "baseline", action_base)

        print(f"[AUTO] {model} | action={action} | reward={reward}")
        print(f"[BASELINE] action={action_base} | reward={reward_base}")

        time.sleep(SLEEP_TIME)

    except Exception as e:
        print("Loop error:", e)
        time.sleep(3)