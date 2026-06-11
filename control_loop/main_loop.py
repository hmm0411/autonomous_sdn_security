import csv
import os
import time
from typing import Optional, Callable, cast

import requests

from rl_engine.state_builder import StateBuilder
from rl_engine.reward import Reward
from typing import Any, Callable, Optional
from control_loop.controller_client import execute_action
from control_loop.rl_client import get_action
from control_loop.metrics import update_metrics, ensure_metrics_server
from control_loop.state_collector import get_state


STATE_DIM = 8

SLEEP_TIME = float(os.getenv("SLEEP_TIME", "2"))
MODE = os.getenv("MODE", "collect").lower()
# collect | no_defense | rule | rl | rl_twin | rl_llm | full

ACTION_DRY_RUN = os.getenv("ACTION_DRY_RUN", "true").lower() == "true"

LOG_PATH = os.getenv("RUNTIME_LOG", "logs/runtime_eval.csv")
TRANSITION_PATH = os.getenv("TRANSITION_LOG", "logs/transition_log.csv")
MODEL_TYPE = os.getenv("MODEL_TYPE", "dqn").lower()

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
    f"AUTO CONTROL LOOP STARTED | MODE={MODE} | "
    f"MODEL_TYPE={MODEL_TYPE} | DRY_RUN={ACTION_DRY_RUN}",
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


def rule_action(raw):
    if raw is None:
        return 0

    pps = float(raw.get("packet_rate", 0.0) or 0.0)
    flows = float(raw.get("flow_count", 0.0) or 0.0)
    growth = abs(float(raw.get("flow_growth_rate", 0.0) or 0.0))
    entropy = float(raw.get("src_ip_entropy", 0.0) or 0.0)
    latency = float(raw.get("latency", 0.0) or 0.0)
    loss = float(raw.get("packet_loss", 0.0) or 0.0)
    cpu = float(raw.get("controller_cpu", 0.0) or 0.0)

    anomaly_score = 0

    if pps > 1000:
        anomaly_score += 1
    if flows > 150:
        anomaly_score += 1
    if growth > 30:
        anomaly_score += 1
    if entropy > 2.0:
        anomaly_score += 1
    if latency > 80:
        anomaly_score += 1
    if loss > 0.05:
        anomaly_score += 1
    if cpu > 80:
        anomaly_score += 1

    if anomaly_score >= 4:
        return 1
    if anomaly_score >= 2:
        return 2
    if entropy > 3.0 and flows > 80:
        return 3

    return 0


def choose_action(raw, state):
    if MODE == "no_defense":
        return 0, 0, "none"

    if MODE == "collect":
        import random
        # Random để dataset có đủ các action
        action = random.randint(0, 4)
        return action, action, "random"

    if MODE == "rule":
        action = rule_action(raw)
        return action, action, "rule"

    action_prod, action_staging, model_name = get_action(
        state,
        model_type=MODEL_TYPE,
    )

    return int(action_prod), int(action_staging), str(model_name).lower()


def log_runtime(row):
    init_csv(
        LOG_PATH,
        [
            "ts",
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

        action_prod, action_staging, model_name = choose_action(raw, state)

        reward_prod = reward_calc.calculate(raw, action_prod)
        reward_staging = reward_calc.calculate(raw, action_staging)

        final_action = action_prod

        twin_safe = 1
        pred_latency = 0.0
        pred_loss = 0.0
        gap_latency = 0.0
        gap_loss = 0.0

        if MODE in ("rl_twin", "full") and TWIN is not None:
            try:
                pred = TWIN.simulate(state, final_action)

                pred_latency = float(pred.get("latency", 0.0))
                pred_loss = float(pred.get("packet_loss", 0.0))

                twin_safe = 1 if TWIN_VALIDATE(pred) else 0

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

        if ACTION_DRY_RUN:
            print(f"[DRY_RUN] would execute action={final_action}", flush=True)
        else:
            execute_action(final_action, raw=raw)

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
                    raw.get("attack_type", None),
                )

                print(
                    "[LLM]",
                    str(explanation)[:180].replace("\n", " "),
                    flush=True,
                )

                log_decision(state_dict, final_action, qos, explanation)

            except Exception as e:
                print(f"[LLM_ERROR] {e}", flush=True)

        time.sleep(0.5)
        next_raw = get_state()

        if next_raw is not None:
            if pred_latency:
                gap_latency = abs(float(next_raw.get("latency", 0.0)) - pred_latency)

            if pred_loss:
                gap_loss = abs(float(next_raw.get("packet_loss", 0.0)) - pred_loss)

        log_transition(
            raw,
            final_action,
            next_raw,
            os.getenv("ATTACK_TYPE", "unknown"),
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
            f"[LOOP] mode={MODE} model={model_name} "
            f"action={final_action} reward={reward_prod:.4f} "
            f"twin_safe={twin_safe} gap_lat={gap_latency:.4f}",
            flush=True,
        )

        time.sleep(SLEEP_TIME)

    except Exception as e:
        print(f"[LOOP_ERROR] {e}", flush=True)
        time.sleep(3)