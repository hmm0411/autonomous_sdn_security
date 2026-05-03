import time
import os
import datetime
import numpy as np

from control_loop.controller_client import execute_action
from control_loop.rl_client import get_best_action
from control_loop.state_collector import get_state
from control_loop.metrics import update_metrics
from rl_engine.reward import Reward

from llm.prompt_builder import build_prompt
from llm.llm_service import call_llm
from rl_engine.logger import Logger

STATE_DIM = 9
SLEEP_TIME = 2

reward_calc = Reward()
logger = Logger(log_dir="results/logs")

warmup_counter = 0

print("AUTO MODEL CONTROL LOOP STARTED")


def norm(x, max_val):
    return min(max(x / max_val, 0.0), 1.0)


def build_state(raw):
    return np.array([
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


while True:
    try:
        # =========================
        # 1️⃣ GET CURRENT STATE
        # =========================
        raw_state = get_state()

        if raw_state is None:
            time.sleep(SLEEP_TIME)
            continue

        state = build_state(raw_state)

        print("STATE:", state)

        # Warmup
        warmup_counter += 1
        if warmup_counter < 5:
            print(f"Warmup... ({warmup_counter}/5)")
            time.sleep(SLEEP_TIME)
            continue

        # =========================
        # 2️⃣ DECIDE ACTION
        # =========================
        action, model_name, _ = get_best_action(state)

        print("SELECTED ACTION:", action)

        # =========================
        # 3️⃣ EXECUTE
        # =========================
        execute_action(action)

        # =========================
        # 4️⃣ WAIT ENV RESPONSE
        # =========================
        time.sleep(SLEEP_TIME)

        next_raw_state = get_state()

        if next_raw_state is None:
            continue

        # =========================
        # 5️⃣ COMPUTE REWARD
        # =========================
        reward = reward_calc.calculate(
            prev_state=raw_state,
            action=action,
            next_state=next_raw_state
        )

        print(f"[AUTO] {model_name} | action={action} | reward={reward}")

        update_metrics(state, reward, model_name, action)

        # =========================
        # 6️⃣ LLM EXPLANATION
        # =========================
        if action != 0:
            qos_metrics = {
                "delay": next_raw_state.get("latency", 0),
                "loss": next_raw_state.get("packet_loss", 0),
                "throughput": next_raw_state.get("byte_rate", 0)
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

    except Exception as e:
        print("Loop error:", e)
        time.sleep(3)