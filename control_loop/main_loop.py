import time
import numpy as np

<<<<<<< Updated upstream
from llm.prompt_builder import PromptBuilder
=======

>>>>>>> Stashed changes
from rl_engine.state_builder import StateBuilder
from control_loop.controller_client import execute_action
from rl_engine.reward import Reward

from control_loop.rl_client import get_best_action
from control_loop.metrics import update_metrics
from control_loop.state_collector import get_state

# LLM Cognition Layer 
try:
    from llm.llm_cognition_layer import (
        explain_decision,
        log_decision,
        state_vector_to_dict,
    )
    LLM_ENABLED = True
    print("[LLM] Cognition Layer loaded ✓")
except Exception as e:
    LLM_ENABLED = False
    print(f"[LLM] Cognition Layer disabled: {e}")

STATE_DIM = 9
SLEEP_TIME = 2

# Only call LLM when a defense action is chosen (not no_action), to keep latency low
LLM_TRIGGER_ACTIONS = {1, 2, 3, 4}

state_builder = StateBuilder()
reward_calc = Reward()
#prompt_builder = PromptBuilder()

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

        # ===== LLM COGNITION LAYER =====
<<<<<<< Updated upstream
        # Trigger LLM explanation only for non-trivial defense actions
        if LLM_ENABLED and action in LLM_TRIGGER_ACTIONS:
            try:
                state_dict = state_vector_to_dict(state)
                qos = {
                    "latency":      float(state[4]),
                    "packet_loss":  float(state[5]),
                    "throughput":   None,          # not available from state vector
                }
                attack_context = raw.get("attack_type", None)

                explanation = explain_decision(state_dict, action, qos, attack_context)
                print(f"[LLM]  {explanation[:120].replace(chr(10),' ')}…")

                log_decision(state_dict, action, qos, explanation)

            except Exception as llm_err:
                print(f"[LLM] Error during cognition: {llm_err}")

=======
        print(f"[DEBUG] LLM_ENABLED={LLM_ENABLED}")
        print(f"[DEBUG] action={action}")

        if LLM_ENABLED and action in LLM_TRIGGER_ACTIONS:
            try:
                state_dict = state_vector_to_dict(state)

                qos = {
                    "latency": float(state[4]),
                    "packet_loss": float(state[5]),
                    "throughput": None,
                }

                attack_context = raw.get("attack_type", None)

                print("[DEBUG] Calling explain_decision")

                explanation = explain_decision(
                    state_dict,
                    action,
                    qos,
                    attack_context
                )

                print("[DEBUG] LLM returned")

                print("\nLLM EXPLANATION")
                print(explanation)

                log_decision(
                    state_dict,
                    action,
                    qos,
                    explanation
                )

            except Exception as llm_err:
                print(f"[LLM] Error during cognition: {llm_err}")
>>>>>>> Stashed changes
        time.sleep(SLEEP_TIME)

    except Exception as e:
        print("Loop error:", e)
        time.sleep(3)