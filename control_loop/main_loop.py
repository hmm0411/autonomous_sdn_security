import time
import os
import datetime
import numpy as np
import csv

from control_loop.controller_client import execute_action
from control_loop.rl_client import get_best_action
from control_loop.state_collector import get_state
from control_loop.metrics import update_metrics
from rl_engine.reward import Reward

from llm.prompt_builder import build_prompt
from llm.llm_service import call_llm
from rl_engine.logger import Logger

from digital_twin.twin_validation import TwinValidator

STATE_DIM = 9
SLEEP_TIME = 2

reward_calc = Reward()
twin = TwinValidator()
logger = Logger(log_dir="results/logs")

warmup_counter = 0
mode = os.getenv("MODE", "rl_twin")

print("AUTO MODEL CONTROL LOOP STARTED")

def norm(x, max_val):
    return min(max(x / max_val, 0.0), 1.0)


def build_state(raw):
    return np.array([
        norm(raw["packet_rate"], 20000),       # 0
        norm(raw["byte_rate"], 500000),       # 1
        norm(raw["flow_count"], 100),         # 2 (flow_ratio)
        np.clip(raw["src_ip_entropy"], 0, 1), # 3 (entropy)
        norm(raw["latency"], 100),            # 4
        np.clip(raw["packet_loss"], 0, 1),    # 5
        np.clip(raw["queue_length"], 0, 1),   # 6
        np.clip(raw["controller_cpu"], 0, 1), # 7
        raw.get("previous_action", 0)         # 8
    ], dtype=np.float32)

def log_transition(state, action, next_state):
    os.makedirs("logs", exist_ok=True)
    file_path = "logs/transition_log.csv"

    header = (
        [f"s{i}" for i in range(len(state))] +
        ["action"] +
        [f"next_s{i}" for i in range(len(next_state))]
    )

    row = list(state) + [action] + list(next_state)

    write_header = not os.path.exists(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


while True:
    try:
        # =========================
        # GET CURRENT STATE
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
        # DECIDE ACTION
        # =========================
        if mode == "no_defense":
            action = 0
            model_name = "NO_DEFENSE"

        elif mode == "rule":
            action = 1 if state[2] > 0.6 else 0
            model_name = "RULE"

        else:
            action, model_name, pressure = get_best_action(state)

            if mode == "rl_twin":
                predicted_next = twin.predict_next_state(state, action)

                if not twin.is_safe(predicted_next):
                    print("Twin REJECT → fallback")
                    action = 0
                else:
                    print("Twin ACCEPT")

        # =========================
        # EXECUTE
        # =========================
        execute_action(action)

        time.sleep(SLEEP_TIME)

        next_raw_state = get_state()
        if next_raw_state is None:
            continue

        next_state = build_state(next_raw_state)

        # =========================
        # WAIT ENV RESPONSE
        # =========================
        if mode == "rl_twin":
            gap = np.mean((predicted_next - next_state) ** 2)

            with open("logs/gap_log.csv", "a") as f:
                f.write(f"{gap},{action}\n")
            print(f"Predicted next state: {predicted_next}")

        # =========================
        # COMPUTE REWARD
        # =========================
        reward = reward_calc.calculate(
            prev_state=raw_state,
            action=action,
            next_state=next_raw_state
        )

        print(f"[AUTO] {model_name} | action={action} | reward={reward}")

        update_metrics(state, reward, model_name, action)

        # =========================
        # LLM EXPLANATION
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