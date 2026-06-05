import csv
import os
import time
from typing import Any, Callable, Optional, cast

import requests

from rl_engine.state_builder import StateBuilder
from rl_engine.reward import Reward

from control_loop.controller_client import execute_action
from control_loop.rl_client import get_action
from control_loop.metrics import update_metrics, ensure_metrics_server
from control_loop.state_collector import get_state


STATE_DIM = 8

SLEEP_TIME = float(os.getenv("SLEEP_TIME", "2"))
MODE = os.getenv("MODE", "collect").lower()
MODEL_TYPE = os.getenv("MODEL_TYPE", "dqn").lower()

ACTION_DRY_RUN = os.getenv("ACTION_DRY_RUN", "true").lower() == "true"

# Bật fallback rule khi RL bị bias action=0.
# Đây là safety guard/hybrid layer, giúp hệ thống vẫn phản ứng khi model chưa học tốt.
RULE_FALLBACK = os.getenv("RULE_FALLBACK", "true").lower() == "true"

LOG_PATH = os.getenv("RUNTIME_LOG", "logs/runtime_eval.csv")
TRANSITION_PATH = os.getenv("TRANSITION_LOG", "logs/transition_log.csv")
ATTACK_TYPE = os.getenv("ATTACK_TYPE", "unknown")

ONOS_HEALTH = os.getenv(
    "ONOS_HEALTH",
    "http://controller:8181/onos/v1/flows"
)

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

state_builder = StateBuilder(normalize=True)
reward_calc = Reward()

ensure_metrics_server(9100)

print(
    f"AUTO CONTROL LOOP STARTED | MODE={MODE} | MODEL_TYPE={MODEL_TYPE} | "
    f"DRY_RUN={ACTION_DRY_RUN} | RULE_FALLBACK={RULE_FALLBACK} | "
    f"ATTACK_TYPE={ATTACK_TYPE}",
    flush=True,
)


# =========================================================
# LLM Cognition Layer
# =========================================================
LLM_ENABLED = False
explain_decision: Optional[Callable[..., Any]] = None
log_decision: Optional[Callable[..., Any]] = None
state_vector_to_dict: Optional[Callable[..., Any]] = None

try:
    from llm.llm_cognition_layer import (
        explain_decision as _explain_decision,
        log_decision as _log_decision,
        state_vector_to_dict as _state_vector_to_dict,
    )

    explain_decision = _explain_decision
    log_decision = _log_decision
    state_vector_to_dict = _state_vector_to_dict

    LLM_ENABLED = True
    print("[LLM] enabled", flush=True)

except Exception as e:
    print(f"[LLM] disabled: {e}", flush=True)


# =========================================================
# Digital Twin
# =========================================================
TWIN: Any = None
TWIN_VALIDATE: Optional[Callable[[dict], bool]] = None

if MODE in ("rl_twin", "full"):
    try:
        from digital_twin.twin import DigitalTwin
        from digital_twin.safety import validate

        surrogate_path = os.getenv(
            "SURROGATE_MODEL",
            "models/surrogate_model.pkl"
        )

        TWIN = DigitalTwin(surrogate_path)
        TWIN_VALIDATE = validate

        print(f"[TWIN] enabled | model={surrogate_path}", flush=True)

    except Exception as e:
        print(f"[TWIN] disabled: {e}", flush=True)
        TWIN = None
        TWIN_VALIDATE = None


# =========================================================
# Helper
# =========================================================
def wait_http_service(name, url, auth=None):
    while True:
        try:
            res = requests.get(url, auth=auth, timeout=3)
            if res.status_code in (200, 401):
                print(f"[READY] {name}", flush=True)
                return
        except Exception as e:
            print(f"[WAIT] {name}: {e}", flush=True)

        time.sleep(5)


def init_csv(path, header):
    parent = os.path.dirname(path)

    if parent:
        os.makedirs(parent, exist_ok=True)

    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def threat_score(raw):
    """
    Threat score dùng cho debug, rule và fallback.
    State raw:
    packet_rate, byte_rate, flow_count, flow_growth_rate,
    src_ip_entropy, latency, packet_loss, controller_cpu
    """
    if raw is None:
        return 0.0

    pps = float(raw.get("packet_rate", 0.0) or 0.0)
    flows = float(raw.get("flow_count", 0.0) or 0.0)
    growth = abs(float(raw.get("flow_growth_rate", 0.0) or 0.0))
    entropy = float(raw.get("src_ip_entropy", 0.0) or 0.0)
    latency = float(raw.get("latency", 0.0) or 0.0)
    loss = float(raw.get("packet_loss", 0.0) or 0.0)
    cpu = float(raw.get("controller_cpu", 0.0) or 0.0)

    score = 0.0

    if pps > 500:
        score += 1.0
    if pps > 2000:
        score += 1.0

    if flows > 50:
        score += 1.0
    if flows > 150:
        score += 1.0

    if growth > 10:
        score += 1.0
    if growth > 50:
        score += 1.0

    if entropy > 1.5:
        score += 1.0

    if latency > 50:
        score += 1.0
    if latency > 120:
        score += 1.0

    if loss > 0.02:
        score += 1.0
    if loss > 0.10:
        score += 1.0

    if cpu > 60:
        score += 1.0

    return score


def rule_action(raw):
    """
    Rule-based baseline đã hạ threshold để demo/evaluation thấy phản ứng.
    """
    if raw is None:
        return 0

    pps = float(raw.get("packet_rate", 0.0) or 0.0)
    flows = float(raw.get("flow_count", 0.0) or 0.0)
    growth = abs(float(raw.get("flow_growth_rate", 0.0) or 0.0))
    entropy = float(raw.get("src_ip_entropy", 0.0) or 0.0)
    latency = float(raw.get("latency", 0.0) or 0.0)
    loss = float(raw.get("packet_loss", 0.0) or 0.0)
    cpu = float(raw.get("controller_cpu", 0.0) or 0.0)

    score = threat_score(raw)

    # Severe case: ưu tiên block.
    if pps > 5000 or loss > 0.20 or latency > 200 or growth > 200:
        return 1

    # Flow overflow/scan: redirect hoặc block nhẹ.
    if growth > 50 or (entropy > 2.0 and flows > 50):
        return 3

    # DDoS/traffic overload: limit bandwidth.
    if pps > 500 or latency > 50 or loss > 0.02 or cpu > 60:
        return 2

    if score >= 3:
        return 2

    return 0


def choose_action(raw, state):
    """
    Return:
      action_prod, action_staging, model_name, rl_action_raw, rule_action_raw
    """
    r_action = rule_action(raw)

    if MODE in ("collect", "no_defense"):
        return 0, 0, "none", 0, r_action

    if MODE == "rule":
        return r_action, r_action, "rule", 0, r_action

    action_prod, action_staging, model_name = get_action(
        state,
        model_type=MODEL_TYPE,
    )

    rl_action = int(action_prod)
    action_prod = int(action_prod)
    action_staging = int(action_staging)
    model_name = str(model_name).lower()

    # Hybrid safety guard:
    # Nếu RL chọn no_action nhưng rule thấy threat rõ ràng, dùng rule để tránh action toàn 0.
    if RULE_FALLBACK and MODE in ("rl", "rl_twin", "rl_llm", "full"):
        score = threat_score(raw)

        if action_prod == 0 and r_action != 0 and score >= 2:
            print(
                f"[RULE_FALLBACK] RL action=0 but threat_score={score:.2f}; "
                f"use rule_action={r_action}",
                flush=True,
            )
            action_prod = r_action
            action_staging = r_action
            model_name = f"{model_name}_guarded"

    return action_prod, action_staging, model_name, rl_action, r_action


def log_runtime(row):
    init_csv(
        LOG_PATH,
        [
            "ts",
            "attack_type",
            "mode",
            "model",
            "packet_rate",
            "byte_rate",
            "flow_count",
            "flow_growth_rate",
            "src_ip_entropy",
            "latency",
            "packet_loss",
            "controller_cpu",
            "threat_score",
            "rl_action",
            "rule_action",
            "action",
            "reward",
            "reward_staging",
            "twin_safe",
            "pred_latency",
            "pred_loss",
            "gap_latency",
            "gap_loss",
        ],
    )

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def log_transition(prev_raw, action, next_raw, attack_type="unknown"):
    init_csv(
        TRANSITION_PATH,
        [
            "packet_rate",
            "byte_rate",
            "flow_count",
            "flow_growth_rate",
            "src_ip_entropy",
            "latency",
            "packet_loss",
            "controller_cpu",
            "action",
            "next_latency",
            "next_packet_loss",
            "attack_type",
        ],
    )

    if prev_raw is None or next_raw is None:
        return

    with open(TRANSITION_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                prev_raw.get("packet_rate", 0.0),
                prev_raw.get("byte_rate", 0.0),
                prev_raw.get("flow_count", 0.0),
                prev_raw.get("flow_growth_rate", 0.0),
                prev_raw.get("src_ip_entropy", 0.0),
                prev_raw.get("latency", 0.0),
                prev_raw.get("packet_loss", 0.0),
                prev_raw.get("controller_cpu", 0.0),
                int(action),
                next_raw.get("latency", 0.0),
                next_raw.get("packet_loss", 0.0),
                attack_type,
            ]
        )


# =========================================================
# Wait dependencies
# =========================================================
wait_http_service(
    "ONOS",
    ONOS_HEALTH,
    auth=(
        os.getenv("ONOS_USER", "onos"),
        os.getenv("ONOS_PASS", "rocks"),
    ),
)

if MODE in ("rl", "rl_twin", "rl_llm", "full"):
    if MODEL_TYPE == "ppo":
        wait_http_service("RL PPO", "http://rl-serving-ppo:8001/health")
    else:
        wait_http_service("RL DQN", "http://rl-serving-dqn:8000/health")


# =========================================================
# Main loop
# =========================================================
while True:
    try:
        raw = get_state()

        if raw is None:
            time.sleep(SLEEP_TIME)
            continue

        state = state_builder.build(raw)

        if len(state) != STATE_DIM:
            print(
                f"[INVALID_STATE] len={len(state)} expected={STATE_DIM} state={state}",
                flush=True,
            )
            time.sleep(SLEEP_TIME)
            continue

        (
            action_prod,
            action_staging,
            model_name,
            rl_action_raw,
            rule_action_raw,
        ) = choose_action(raw, state)

        reward_prod = reward_calc.calculate(raw, action_prod)
        reward_staging = reward_calc.calculate(raw, action_staging)

        final_action = int(action_prod)

        score = threat_score(raw)

        twin_safe = 1
        pred_latency = 0.0
        pred_loss = 0.0
        gap_latency = 0.0
        gap_loss = 0.0

        print(
            f"[ACTION_DEBUG] attack={ATTACK_TYPE} mode={MODE} model={model_name} "
            f"rl={rl_action_raw} rule={rule_action_raw} final={final_action} "
            f"score={score:.2f} pps={raw.get('packet_rate', 0.0):.2f} "
            f"flows={raw.get('flow_count', 0.0):.2f} "
            f"growth={raw.get('flow_growth_rate', 0.0):.2f} "
            f"lat={raw.get('latency', 0.0):.2f} "
            f"loss={raw.get('packet_loss', 0.0):.4f}",
            flush=True,
        )

        # =========================
        # Digital Twin validation
        # =========================
        if MODE in ("rl_twin", "full") and TWIN is not None and TWIN_VALIDATE is not None:
            try:
                pred = TWIN.simulate(state, final_action)

                pred_latency = float(pred.get("latency", 0.0))
                pred_loss = float(pred.get("packet_loss", 0.0))

                validate_fn = cast(Callable[[dict], bool], TWIN_VALIDATE)
                twin_safe = 1 if validate_fn(pred) else 0

                if twin_safe == 0:
                    print(
                        f"[TWIN_REJECT] action={final_action} "
                        f"pred_latency={pred_latency:.4f} "
                        f"pred_loss={pred_loss:.4f} -> fallback no_action",
                        flush=True,
                    )
                    final_action = 0

            except Exception as e:
                print(f"[TWIN_ERROR] {e}", flush=True)
                twin_safe = 0
                final_action = 0

        # =========================
        # Apply action
        # =========================
        if ACTION_DRY_RUN:
            print(f"[DRY_RUN] would execute action={final_action}", flush=True)
        else:
            execute_action(final_action, raw=raw)

        # =========================
        # LLM explanation
        # =========================
        if (
            MODE in ("rl_llm", "full")
            and LLM_ENABLED
            and explain_decision is not None
            and log_decision is not None
            and state_vector_to_dict is not None
            and final_action in (1, 2, 3, 4)
        ):
            try:
                state_dict = state_vector_to_dict(state)

                qos = {
                    "latency": raw.get("latency", 0.0),
                    "packet_loss": raw.get("packet_loss", 0.0),
                    "throughput": raw.get("byte_rate", 0.0),
                }

                explanation = explain_decision(
                    state_dict,
                    final_action,
                    qos,
                    raw.get("attack_type", ATTACK_TYPE),
                )

                print(
                    "[LLM]",
                    str(explanation)[:180].replace("\n", " "),
                    flush=True,
                )

                log_decision(state_dict, final_action, qos, explanation)

            except Exception as e:
                print(f"[LLM_ERROR] {e}", flush=True)

        # =========================
        # Observe next state
        # =========================
        time.sleep(0.5)
        next_raw = get_state()

        if next_raw is not None:
            # Không dùng if pred_latency nữa, vì pred=0 cũng cần tính gap.
            gap_latency = abs(float(next_raw.get("latency", 0.0)) - pred_latency)
            gap_loss = abs(float(next_raw.get("packet_loss", 0.0)) - pred_loss)

        log_transition(
            raw,
            final_action,
            next_raw,
            ATTACK_TYPE,
        )

        update_metrics(
            state=state,
            reward_prod=reward_prod,
            reward_staging=reward_staging,
            model_type_str=model_name,
            action=final_action,
            twin_safe=twin_safe,
            twin_gap_latency=gap_latency,
        )

        log_runtime(
            [
                time.time(),
                ATTACK_TYPE,
                MODE,
                model_name,
                raw.get("packet_rate", 0.0),
                raw.get("byte_rate", 0.0),
                raw.get("flow_count", 0.0),
                raw.get("flow_growth_rate", 0.0),
                raw.get("src_ip_entropy", 0.0),
                raw.get("latency", 0.0),
                raw.get("packet_loss", 0.0),
                raw.get("controller_cpu", 0.0),
                score,
                rl_action_raw,
                rule_action_raw,
                final_action,
                reward_prod,
                reward_staging,
                twin_safe,
                pred_latency,
                pred_loss,
                gap_latency,
                gap_loss,
            ]
        )

        print(
            f"[LOOP] attack={ATTACK_TYPE} mode={MODE} model={model_name} "
            f"action={final_action} reward={reward_prod:.4f} "
            f"twin_safe={twin_safe} gap_lat={gap_latency:.4f}",
            flush=True,
        )

        time.sleep(SLEEP_TIME)

    except Exception as e:
        print(f"[LOOP_ERROR] {e}", flush=True)
        time.sleep(3)