import csv
import os
import time
from typing import Any, Callable, Optional

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
# Supported modes:
# collect        : baseline collect, action always 0
# collect_random : collect transition data with random actions for Digital Twin
# no_defense     : no defense baseline, action always 0
# rule           : rule-based heuristic
# rl             : RL-only
# rl_twin        : RL + Digital Twin safety validation
# rl_llm         : RL + LLM explanation
# full           : RL + Twin + LLM
# hybrid         : DQN/PPO hybrid arbitration

ACTION_DRY_RUN = os.getenv("ACTION_DRY_RUN", "true").lower() == "true"

LOG_PATH = os.getenv("RUNTIME_LOG", "logs/runtime_eval.csv")
TRANSITION_PATH = os.getenv("TRANSITION_LOG", "logs/transition_log.csv")
MODEL_TYPE = os.getenv("MODEL_TYPE", "dqn").lower()

ONOS_HEALTH = os.getenv(
    "ONOS_HEALTH",
    "http://controller:8181/onos/v1/flows",
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
# Helpers
# =========================================================
def raw_to_state_vector(raw: dict) -> list[float]:
    """
    Raw state vector 8 chiều dùng cho:
    - logging
    - Prometheus metric theo giá trị thật
    - Digital Twin nếu surrogate được train từ transition_log.csv raw
    """
    return [
        float(raw.get("packet_rate", 0.0) or 0.0),
        float(raw.get("byte_rate", 0.0) or 0.0),
        float(raw.get("flow_count", 0.0) or 0.0),
        float(raw.get("flow_growth_rate", 0.0) or 0.0),
        float(raw.get("src_ip_entropy", 0.0) or 0.0),
        float(raw.get("latency", 0.0) or 0.0),
        float(raw.get("packet_loss", 0.0) or 0.0),
        float(raw.get("controller_cpu", 0.0) or 0.0),
    ]

def pressure_score(raw: dict) -> float:
    """
    Pressure score cho hybrid arbitration.
    Giá trị càng cao thì hệ thống càng bất thường.
    Dùng DQN khi pressure cao, PPO khi pressure thấp.
    """
    pps = float(raw.get("packet_rate", 0.0) or 0.0)
    flows = float(raw.get("flow_count", 0.0) or 0.0)
    growth = abs(float(raw.get("flow_growth_rate", 0.0) or 0.0))
    entropy = float(raw.get("src_ip_entropy", 0.0) or 0.0)
    latency = float(raw.get("latency", 0.0) or 0.0)
    loss = float(raw.get("packet_loss", 0.0) or 0.0)
    cpu = float(raw.get("controller_cpu", 0.0) or 0.0)

    score = 0.0
    score += min(pps / 5000.0, 1.0) * 0.25
    score += min(flows / 200.0, 1.0) * 0.15
    score += min(growth / 100.0, 1.0) * 0.15
    score += min(entropy / 8.0, 1.0) * 0.15
    score += min(latency / 300.0, 1.0) * 0.10
    score += min(loss if loss <= 1.0 else loss / 100.0, 1.0) * 0.10
    score += min(cpu / 100.0, 1.0) * 0.10

    return float(score)


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

        TWIN = DigitalTwin(
            os.getenv("SURROGATE_MODEL", "models/surrogate_model.pkl")
        )
        TWIN_VALIDATE = validate
        print("[TWIN] enabled", flush=True)

    except Exception as e:
        print(f"[TWIN] disabled: {e}", flush=True)
        TWIN = None
        TWIN_VALIDATE = None


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


def _f(raw, key, default=0.0):
    try:
        return float(raw.get(key, default) or default)
    except Exception:
        return float(default)


def threat_score(raw):
    """
    Tính điểm bất thường từ raw state runtime.
    Hàm này dùng cho rule-based, action guard và benchmark.
    """
    if raw is None:
        return 0

    pps = _f(raw, "packet_rate")
    bps = _f(raw, "byte_rate")
    flows = _f(raw, "flow_count")
    growth = abs(_f(raw, "flow_growth_rate"))
    entropy = _f(raw, "src_ip_entropy")
    latency = _f(raw, "latency")
    loss = _f(raw, "packet_loss")
    cpu = _f(raw, "controller_cpu")

    score = 0

    # Packet/byte rate
    if pps >= 2000:
        score += 3
    elif pps >= 500:
        score += 2
    elif pps >= 100:
        score += 1

    if bps >= 100000:
        score += 2
    elif bps >= 10000:
        score += 1

    # Flow pressure
    if flows >= 50:
        score += 2
    elif flows >= 20:
        score += 1

    if growth >= 20:
        score += 2
    elif growth >= 3:
        score += 1

    # Source diversity / spoofing indicator
    if entropy >= 0.7:
        score += 2
    elif entropy >= 0.4:
        score += 1

    # QoS degradation
    if latency >= 100:
        score += 3
    elif latency >= 20:
        score += 2
    elif latency >= 8:
        score += 1

    if loss >= 0.05:
        score += 3
    elif loss >= 0.005:
        score += 2
    elif loss >= 0.001:
        score += 1

    # Controller overload
    if cpu >= 80:
        score += 3
    elif cpu >= 40:
        score += 2
    elif cpu >= 18:
        score += 1

    return score

def is_clearly_normal(raw):
    """
    Normal guard:
    Không nên chỉ dựa vào flow_count vì ONOS có thể giữ flow cũ.
    Tập trung vào packet_rate, latency, loss, cpu, flow_growth.
    """
    if raw is None:
        return True

    pps = _f(raw, "packet_rate")
    growth = abs(_f(raw, "flow_growth_rate"))
    latency = _f(raw, "latency")
    loss = _f(raw, "packet_loss")
    cpu = _f(raw, "controller_cpu")
    entropy = _f(raw, "src_ip_entropy")

    return (
        pps < 80
        and growth < 3
        and entropy < 0.4
        and latency < 8
        and loss < 0.001
        and cpu < 18
    )

def rule_action(raw):
    """
    Rule-based baseline đã chỉnh theo metric thực tế bạn đang thấy.
    Không dùng ngưỡng quá cao nữa, vì log runtime có nhiều attack ở mức
    packet_rate 100-500 nhưng vẫn đã làm latency/loss tăng.
    """
    if raw is None:
        return 0

    pps = _f(raw, "packet_rate")
    bps = _f(raw, "byte_rate")
    flows = _f(raw, "flow_count")
    growth = abs(_f(raw, "flow_growth_rate"))
    entropy = _f(raw, "src_ip_entropy")
    latency = _f(raw, "latency")
    loss = _f(raw, "packet_loss")
    cpu = _f(raw, "controller_cpu")

    score = threat_score(raw)

    # Mạng bình thường thì không can thiệp
    if is_clearly_normal(raw):
        return 0

    # Tấn công/overload rất nặng: block
    if (
        pps >= 2000
        or latency >= 100
        or loss >= 0.05
        or cpu >= 80
        or score >= 6
    ):
        return 1  # block_suspicious_flow

    # Packet-in/flow pressure: limit bandwidth
    if (
        pps >= 100
        or bps >= 10000
        or latency >= 8
        or loss >= 0.001
        or cpu >= 18
        or flows >= 16
        or score >= 3
    ):
        return 2  # limit_bandwidth

    # Flow growth/scan/spread traffic: redirect
    if growth >= 3 or entropy >= 0.4:
        return 3  # redirect_traffic

    return 0


def apply_action_guard(raw, proposed_action, mode):
    """
    Lớp bảo vệ quyết định:
    - collect/no_defense: luôn giữ 0 để đúng baseline.
    - collect_random: giữ random action để thu transition đa dạng cho Twin.
    - normal traffic: ép action về 0 để tránh overreaction.
    - attack rõ ràng nhưng RL trả 0: override bằng rule_action.
    """
    try:
        proposed_action = int(proposed_action)
    except Exception:
        proposed_action = 0

    mode = str(mode).lower()

    if mode in ("collect", "no_defense"):
        return 0

    if mode == "collect_random":
        return proposed_action

    if is_clearly_normal(raw):
        if proposed_action != 0:
            print(
                f"[ACTION_GUARD] normal traffic detected, override {proposed_action} -> 0",
                flush=True,
            )
        return 0

    fallback_action = rule_action(raw)

    if proposed_action == 0 and fallback_action != 0:
        print(
            f"[ACTION_GUARD] attack detected, override 0 -> {fallback_action} | "
            f"threat_score={threat_score(raw)} raw={raw}",
            flush=True,
        )
        return fallback_action

    return proposed_action


def choose_action(raw, state):
    """
    Return:
    action_prod, action_staging, model_name
    """
    if MODE in ("collect", "no_defense"):
        return 0, 0, "none"

    if MODE == "collect_random":
        import random

        action = random.randint(0, 4)
        return action, action, "random"

    if MODE == "rule":
        action = rule_action(raw)
        return action, action, "rule"

    if MODE == "hybrid":
        threshold = float(os.getenv("HYBRID_PRESSURE_THRESHOLD", "0.45"))
        selected_model = "dqn" if pressure_score(raw) >= threshold else "ppo"

        action_prod, action_staging, model_name = get_action(
            state,
            model_type=selected_model,
        )
        return int(action_prod), int(action_staging), f"hybrid_{model_name}".lower()

    if MODE in ("rl", "rl_twin", "rl_llm", "full"):
        action_prod, action_staging, model_name = get_action(
            state,
            model_type=MODEL_TYPE,
        )
        return int(action_prod), int(action_staging), str(model_name).lower()

    return 0, 0, "unknown"


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
            "attack_type",
            "intensity",
            "run_id",
        ],
    )

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def log_transition(
    prev_raw,
    action,
    next_raw,
    attack_type="unknown",
    intensity="medium",
    run_id="0",
):
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
            "intensity",
            "run_id",
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
                intensity,
                run_id,
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

if MODE in ("rl", "rl_twin", "rl_llm", "full", "hybrid"):
    if MODE == "hybrid":
        wait_http_service("RL DQN", "http://rl-serving-dqn:8000/health")
        wait_http_service("RL PPO", "http://rl-serving-ppo:8001/health")
    elif MODEL_TYPE == "ppo":
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
        raw_state = raw_to_state_vector(raw)

        if len(state) != STATE_DIM:
            print(
                f"[INVALID_STATE] len={len(state)} expected={STATE_DIM} state={state}",
                flush=True,
            )
            time.sleep(SLEEP_TIME)
            continue

        action_prod, action_staging, model_name = choose_action(raw, state)

        final_action = apply_action_guard(
            raw=raw,
            proposed_action=action_prod,
            mode=MODE,
        )

        reward_staging = reward_calc.calculate(raw, action_staging)

        if MODE in ("rl", "rl_twin", "rl_llm", "full", "hybrid") and is_clearly_normal(raw):
            print(
                f"[NORMAL_GUARD] raw metrics look normal, override action {final_action} -> 0",
                flush=True,
            )
            final_action = 0

        twin_safe = 1
        pred_latency = 0.0
        pred_loss = 0.0
        gap_latency = 0.0
        gap_loss = 0.0

        if MODE in ("rl_twin", "full") and TWIN is not None:
            try:
                pred = TWIN.simulate(raw_state, final_action)

                pred_latency = float(pred.get("latency", 0.0))
                pred_loss = float(pred.get("packet_loss", 0.0))

                if TWIN_VALIDATE is not None:
                    twin_safe = 1 if TWIN_VALIDATE(pred) else 0
                else:
                    twin_safe = 0

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

        reward_final = reward_calc.calculate(raw, final_action)

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
                state_dict = state_vector_to_dict(raw_state)

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

        attack_type = os.getenv("ATTACK_TYPE", "unknown")
        intensity = os.getenv("ATTACK_INTENSITY", "medium")
        run_id = os.getenv("RUN_ID", "0")

        log_transition(
            raw,
            final_action,
            next_raw,
            attack_type,
            intensity,
            run_id,
        )

        update_metrics(
            state=raw_state,
            reward_prod=reward_final,
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
                reward_final,
                reward_staging,
                twin_safe,
                pred_latency,
                pred_loss,
                gap_latency,
                gap_loss,
                attack_type,
                intensity,
                run_id,
            ]
        )

        print(
            f"[LOOP] mode={MODE} model={model_name} "
            f"action={final_action} reward={reward_final:.4f} "
            f"twin_safe={twin_safe} gap_lat={gap_latency:.4f}",
            flush=True,
        )

        time.sleep(SLEEP_TIME)

    except Exception as e:
        print(f"[LOOP_ERROR] {e}", flush=True)
        time.sleep(3)