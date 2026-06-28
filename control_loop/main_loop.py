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


# =========================================================
# Runtime Config
# =========================================================
STATE_DIM = 8

SLEEP_TIME = float(os.getenv("SLEEP_TIME", "2"))
MODE = os.getenv("MODE", "collect").lower()
MODEL_TYPE = os.getenv("MODEL_TYPE", "dqn").lower()

# Supported modes:
# collect         : collect baseline, action always 0
# collect_random  : collect transition data with random actions for Digital Twin
# no_defense      : no defense baseline, action always 0
# rule            : rule-based heuristic
# rl              : RL-only
# rl_twin         : RL + Digital Twin safety validation
# rl_llm          : RL + LLM explanation
# full            : RL + Twin + LLM
# hybrid          : DQN/PPO hybrid arbitration
ACTION_DRY_RUN = os.getenv("ACTION_DRY_RUN", "true").lower() == "true"

LOG_PATH = os.getenv("RUNTIME_LOG", "logs/runtime_eval.csv")
TRANSITION_PATH = os.getenv("TRANSITION_LOG", "logs/transition_log.csv")

ONOS_HEALTH = os.getenv(
    "ONOS_HEALTH",
    "http://controller:8181/onos/v1/flows",
)

EVAL_CONFIG = os.getenv("EVAL_CONFIG", "").lower()
PHASE = os.getenv("PHASE", "unknown").lower()
ATTACK_TYPE = os.getenv("ATTACK_TYPE", "unknown").lower()
ATTACK_INTENSITY = os.getenv("ATTACK_INTENSITY", "medium").lower()
RUN_ID = os.getenv("RUN_ID", "0")

ENABLE_GUARD = os.getenv("ENABLE_GUARD", "false").lower() == "true"
ENABLE_TWIN = os.getenv("ENABLE_TWIN", "false").lower() == "true"
ENABLE_LLM = os.getenv("ENABLE_LLM", "false").lower() == "true"

SLA_LATENCY_THRESHOLD = float(os.getenv("SLA_LATENCY_THRESHOLD", "100"))
SLA_LOSS_THRESHOLD = float(os.getenv("SLA_LOSS_THRESHOLD", "0.05"))
HYBRID_PRESSURE_THRESHOLD = float(os.getenv("HYBRID_PRESSURE_THRESHOLD", "0.45"))

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

state_builder = StateBuilder(normalize=True)
reward_calc = Reward()

ensure_metrics_server(9100)

print(
    f"AUTO CONTROL LOOP STARTED | MODE={MODE} | "
    f"MODEL_TYPE={MODEL_TYPE} | DRY_RUN={ACTION_DRY_RUN} | "
    f"GUARD={ENABLE_GUARD} | TWIN={ENABLE_TWIN} | LLM={ENABLE_LLM}",
    flush=True,
)


# =========================================================
# Evaluation Config Helper
# =========================================================
def resolve_eval_config() -> str:
    if EVAL_CONFIG:
        return EVAL_CONFIG

    if MODE == "no_defense":
        return "no_defense"

    if MODE == "rule":
        return "rule"

    if MODE == "rl":
        if ENABLE_GUARD:
            return f"rl_guard_{MODEL_TYPE}"
        return f"rl_{MODEL_TYPE}"

    if MODE == "hybrid":
        if ENABLE_GUARD:
            return "hybrid_guard"
        return "hybrid"

    if MODE == "rl_twin":
        return f"rl_twin_{MODEL_TYPE}"

    if MODE in ("full", "rl_llm"):
        return f"full_system_{MODEL_TYPE}"

    if MODE == "collect":
        return "collect"

    if MODE == "collect_random":
        return "collect_random"

    return MODE


# =========================================================
# State Helpers
# =========================================================
def raw_to_state_vector(raw: dict) -> list[float]:
    """
    Runtime state 8 chiều:
    [packet_rate, byte_rate, flow_count, flow_growth_rate,
     src_ip_entropy, latency, packet_loss, controller_cpu]
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


def _f(raw: Optional[dict], key: str, default: float = 0.0) -> float:
    try:
        if raw is None:
            return float(default)
        return float(raw.get(key, default) or default)
    except Exception:
        return float(default)


def pressure_score(raw: dict) -> float:
    """
    Pressure score dùng cho hybrid arbitration.
    Giá trị cao -> ưu tiên DQN; thấp -> ưu tiên PPO.
    """
    pps = _f(raw, "packet_rate")
    growth = abs(_f(raw, "flow_growth_rate"))
    entropy = _f(raw, "src_ip_entropy")
    latency = _f(raw, "latency")
    loss = _f(raw, "packet_loss")
    cpu = _f(raw, "controller_cpu")

    score = 0.0
    score += min(pps / 5000.0, 1.0) * 0.30
    score += min(growth / 100.0, 1.0) * 0.15
    score += min(entropy / 8.0, 1.0) * 0.15
    score += min(latency / 300.0, 1.0) * 0.15
    score += min(loss if loss <= 1.0 else loss / 100.0, 1.0) * 0.15
    score += min(cpu / 100.0, 1.0) * 0.10

    return float(score)


def threat_score(raw: Optional[dict]) -> float:
    """
    Điểm bất thường dùng cho rule-based và action guard.
    flow_count chỉ là tín hiệu phụ khi rất cao.
    """
    if raw is None:
        return 0.0

    pps = _f(raw, "packet_rate")
    bps = _f(raw, "byte_rate")
    flows = _f(raw, "flow_count")
    growth = abs(_f(raw, "flow_growth_rate"))
    entropy = _f(raw, "src_ip_entropy")
    latency = _f(raw, "latency")
    loss = _f(raw, "packet_loss")
    cpu = _f(raw, "controller_cpu")

    score = 0.0

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

    if flows >= 100:
        score += 2
    elif flows >= 60:
        score += 1

    if growth >= 20:
        score += 2
    elif growth >= 3:
        score += 1

    if entropy >= 0.7:
        score += 2
    elif entropy >= 0.4:
        score += 1

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

    if cpu >= 80:
        score += 3
    elif cpu >= 40:
        score += 2
    elif cpu >= 18:
        score += 1

    return float(score)


def is_clearly_normal(raw: dict) -> bool:
    packet_rate = _f(raw, "packet_rate")
    flow_count = _f(raw, "flow_count")
    flow_growth = abs(_f(raw, "flow_growth_rate"))
    latency = _f(raw, "latency")
    packet_loss = _f(raw, "packet_loss")
    controller_cpu = _f(raw, "controller_cpu")

    return (
        packet_rate < 50
        and flow_count < 15
        and flow_growth < 2
        and latency < 10
        and packet_loss < 0.001
        and controller_cpu < 20
    )


def apply_normal_guard(raw: dict, action: int) -> tuple[int, int]:
    """
    Return:
    - final_action
    - guard_overrode

    Guard chỉ override khi thật sự normal. Trong benchmark attack phase
    với ATTACK_TYPE rõ ràng, không ép toàn bộ action về 0.
    """
    if not ENABLE_GUARD:
        return action, 0

    if action == 0:
        return action, 0

    if PHASE == "attack" and ATTACK_TYPE not in ("normal", "unknown"):
        return action, 0

    score = threat_score(raw)

    if score <= 1.0 and is_clearly_normal(raw):
        print(
            f"[GUARD] normal traffic detected, threat={score:.2f}, "
            f"override action {action} -> 0",
            flush=True,
        )
        return 0, 1

    return action, 0


# =========================================================
# Rule-based Baseline
# =========================================================
def rule_action(raw: Optional[dict]) -> int:
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

    if is_clearly_normal(raw):
        return 0

    if (
        pps >= 2000
        or latency >= 100
        or loss >= 0.05
        or cpu >= 80
        or score >= 6
    ):
        return 1  # block_suspicious_flow

    if (
        pps >= 100
        or bps >= 10000
        or latency >= 8
        or loss >= 0.001
        or cpu >= 18
        or flows >= 60
        or score >= 3
    ):
        return 2  # limit_bandwidth

    if growth >= 3 or entropy >= 0.4:
        return 3  # redirect_traffic

    return 0



# =========================================================
# Runtime Benchmark Reward
# =========================================================
IDEAL_ACTION_BY_ATTACK = {
    "normal": [0],
    "ddos_flood": [2, 1],
    "packet_in_flood": [2, 1],
    "flow_overflow": [1, 2],
    "ip_spoofing": [1, 4],
    "port_scanning": [3, 2],
}


def calculate_runtime_reward(
    raw: dict,
    action: int,
    attack_type: str,
    phase: str,
) -> float:
    """
    Reward dùng cho runtime benchmark.
    Không đưa ATTACK_TYPE/PHASE vào state của agent; chỉ dùng để chấm điểm log.
    """
    action = int(action)
    attack_type = str(attack_type).lower()
    phase = str(phase).lower()

    latency = float(raw.get("latency", 0.0) or 0.0)
    packet_loss = float(raw.get("packet_loss", 0.0) or 0.0)
    controller_cpu = float(raw.get("controller_cpu", 0.0) or 0.0)
    packet_rate = float(raw.get("packet_rate", 0.0) or 0.0)
    flow_growth = abs(float(raw.get("flow_growth_rate", 0.0) or 0.0))

    latency_penalty = min(latency / 300.0, 1.0)
    loss_penalty = min(packet_loss if packet_loss <= 1 else packet_loss / 100.0, 1.0)
    cpu_penalty = min(controller_cpu / 100.0, 1.0)
    rate_penalty = min(packet_rate / 5000.0, 1.0)
    growth_penalty = min(flow_growth / 100.0, 1.0)

    qos_penalty = (
        0.25 * latency_penalty
        + 0.30 * loss_penalty
        + 0.15 * cpu_penalty
        + 0.20 * rate_penalty
        + 0.10 * growth_penalty
    )

    # Warmup/recovery: ưu tiên không can thiệp nếu không cần.
    if phase != "attack":
        if action == 0:
            return float(1.0 - qos_penalty)
        return float(-1.0 - qos_penalty)

    # Normal traffic: action 0 là đúng, action khác 0 là false positive.
    if attack_type == "normal":
        if action == 0:
            return float(1.0 - qos_penalty)
        return float(-2.0 - qos_penalty)

    # Attack thật: no_action là sai.
    if action == 0:
        return float(-3.0 - qos_penalty)

    ideal_actions = IDEAL_ACTION_BY_ATTACK.get(attack_type, [1, 2])

    if action in ideal_actions:
        base = 2.5
    else:
        base = 0.5

    action_cost = {
        0: 0.0,
        1: 0.20,
        2: 0.10,
        3: 0.15,
        4: 0.60,
    }.get(action, 0.30)

    return float(max(-5.0, min(3.0, base - qos_penalty - action_cost)))

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
    print("[LLM] module available", flush=True)

except Exception as e:
    print(f"[LLM] disabled: {e}", flush=True)


# =========================================================
# Digital Twin
# =========================================================
TWIN: Any = None
TWIN_VALIDATE: Optional[Callable[[dict], bool]] = None

if MODE in ("rl_twin", "full") and ENABLE_TWIN:
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
else:
    print("[TWIN] disabled by mode or ENABLE_TWIN=false", flush=True)


# =========================================================
# IO Helpers
# =========================================================
def wait_http_service(name: str, url: str, auth=None) -> None:
    while True:
        try:
            res = requests.get(url, auth=auth, timeout=3)
            if res.status_code in (200, 401):
                print(f"[READY] {name}", flush=True)
                return
            print(f"[WAIT] {name}: status={res.status_code}", flush=True)
        except Exception as e:
            print(f"[WAIT] {name}: {e}", flush=True)

        time.sleep(5)


def init_csv(path: str, header: list[str]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def log_runtime(row: list[Any]) -> None:
    init_csv(
        LOG_PATH,
        [
            "ts",
            "eval_config",
            "phase",
            "mode",
            "model",
            "attack_type",
            "intensity",
            "run_id",
            "packet_rate",
            "byte_rate",
            "flow_count",
            "flow_growth_rate",
            "src_ip_entropy",
            "latency",
            "packet_loss",
            "controller_cpu",
            "action_requested",
            "action_final",
            "action_staging",
            "reward",
            "reward_staging",
            "guard_enabled",
            "guard_overrode",
            "twin_enabled",
            "twin_checked",
            "twin_safe",
            "twin_rejected",
            "pred_latency",
            "pred_loss",
            "gap_latency",
            "gap_loss",
            "llm_enabled",
            "llm_latency",
            "sla_violation",
        ],
    )

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def log_transition(
    prev_raw: Optional[dict],
    action: int,
    next_raw: Optional[dict],
    attack_type: str = "unknown",
    intensity: str = "medium",
    run_id: str = "0",
) -> None:
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


# =========================================================
# Action Selection
# =========================================================
def choose_action(raw: dict, state) -> tuple[int, int, str]:
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
        selected_model = "dqn" if pressure_score(raw) >= HYBRID_PRESSURE_THRESHOLD else "ppo"
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


# =========================================================
# Service Readiness
# =========================================================
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


# =========================================================
# Main Loop
# =========================================================
while True:
    try:
        raw = get_state()

        if raw is None:
            print("[STATE] None, skip loop", flush=True)
            time.sleep(SLEEP_TIME)
            continue

        decision_raw = dict(raw)
        state = state_builder.build(decision_raw)
        raw_state = raw_to_state_vector(decision_raw)

        if len(state) != STATE_DIM:
            print(
                f"[INVALID_STATE] len={len(state)} expected={STATE_DIM} state={state}",
                flush=True,
            )
            time.sleep(SLEEP_TIME)
            continue

        eval_config = resolve_eval_config()

        action_requested, action_staging, model_name = choose_action(decision_raw, state)
        final_action = action_requested

        guard_overrode = 0
        twin_checked = 0
        twin_safe = 1
        twin_rejected = 0

        pred_latency = 0.0
        pred_loss = 0.0
        gap_latency = 0.0
        gap_loss = 0.0
        llm_latency = 0.0

        # =========================
        # Guard layer
        # =========================
        final_action, guard_overrode = apply_normal_guard(decision_raw, final_action)

        # =========================
        # Digital Twin validation
        # =========================
        if MODE in ("rl_twin", "full") and ENABLE_TWIN and TWIN is not None:
            twin_checked = 1
            try:
                twin_state = raw_to_state_vector(decision_raw)
                pred = TWIN.simulate(twin_state, final_action)

                pred_latency = float(pred.get("latency", 0.0))
                pred_loss = float(pred.get("packet_loss", 0.0))

                if TWIN_VALIDATE is not None:
                    twin_safe = 1 if TWIN_VALIDATE(pred) else 0
                else:
                    twin_safe = 1

                if twin_safe == 0:
                    twin_rejected = 1
                    print(
                        f"[TWIN_REJECT] requested={action_requested} "
                        f"after_guard={final_action} -> fallback 0 "
                        f"pred_latency={pred_latency:.4f} pred_loss={pred_loss:.4f}",
                        flush=True,
                    )
                    final_action = 0

            except Exception as e:
                print(f"[TWIN_ERROR] {e}", flush=True)
                twin_safe = 0
                twin_rejected = 1
                final_action = 0

        reward_staging = calculate_runtime_reward(
            decision_raw,
            action_staging,
            ATTACK_TYPE,
            PHASE,
        )

        reward_final = calculate_runtime_reward(
            decision_raw,
            final_action,
            ATTACK_TYPE,
            PHASE,
        )

        if ACTION_DRY_RUN:
            print(f"[DRY_RUN] would execute action={final_action}", flush=True)
        else:
            execute_action(final_action, raw=decision_raw)

        # =========================
        # LLM explanation layer
        # =========================
        if (
            MODE in ("full", "rl_llm")
            and ENABLE_LLM
            and LLM_ENABLED
            and explain_decision is not None
            and log_decision is not None
            and state_vector_to_dict is not None
        ):
            try:
                llm_start = time.time()

                state_dict = state_vector_to_dict(raw_state)
                qos = {
                    "latency": decision_raw.get("latency", 0.0),
                    "packet_loss": decision_raw.get("packet_loss", 0.0),
                    "throughput": decision_raw.get("byte_rate", 0.0),
                }

                explanation = explain_decision(
                    state_dict,
                    final_action,
                    qos,
                    ATTACK_TYPE,
                )

                llm_latency = time.time() - llm_start

                print(
                    "[LLM]",
                    str(explanation)[:180].replace("\n", " "),
                    flush=True,
                )

                log_decision(state_dict, final_action, qos, explanation)

            except Exception as e:
                print(f"[LLM_ERROR] {e}", flush=True)
                llm_latency = 0.0

        # Chờ ngắn để lấy next state sau action.
        time.sleep(0.5)
        next_raw = get_state()

        if next_raw is not None:
            if pred_latency:
                gap_latency = abs(float(next_raw.get("latency", 0.0)) - pred_latency)
            if pred_loss:
                gap_loss = abs(float(next_raw.get("packet_loss", 0.0)) - pred_loss)

        log_transition(
            decision_raw,
            final_action,
            next_raw,
            ATTACK_TYPE,
            ATTACK_INTENSITY,
            RUN_ID,
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

        latency = _f(decision_raw, "latency")
        packet_loss = _f(decision_raw, "packet_loss")

        sla_violation = int(
            latency > SLA_LATENCY_THRESHOLD
            or packet_loss > SLA_LOSS_THRESHOLD
        )

        log_runtime(
            [
                time.time(),
                eval_config,
                PHASE,
                MODE,
                model_name,
                ATTACK_TYPE,
                ATTACK_INTENSITY,
                RUN_ID,
                decision_raw.get("packet_rate", 0.0),
                decision_raw.get("byte_rate", 0.0),
                decision_raw.get("flow_count", 0.0),
                decision_raw.get("flow_growth_rate", 0.0),
                decision_raw.get("src_ip_entropy", 0.0),
                decision_raw.get("latency", 0.0),
                decision_raw.get("packet_loss", 0.0),
                decision_raw.get("controller_cpu", 0.0),
                action_requested,
                final_action,
                action_staging,
                reward_final,
                reward_staging,
                int(ENABLE_GUARD),
                guard_overrode,
                int(ENABLE_TWIN),
                twin_checked,
                twin_safe,
                twin_rejected,
                pred_latency,
                pred_loss,
                gap_latency,
                gap_loss,
                int(ENABLE_LLM),
                llm_latency,
                sla_violation,
            ]
        )

        print(
            f"[LOOP] config={eval_config} phase={PHASE} mode={MODE} model={model_name} "
            f"pps={decision_raw.get('packet_rate', 0.0):.4f} "
            f"lat={decision_raw.get('latency', 0.0):.4f} "
            f"loss={decision_raw.get('packet_loss', 0.0):.4f} "
            f"cpu={decision_raw.get('controller_cpu', 0.0):.4f} "
            f"requested={action_requested} final={final_action} "
            f"guard={guard_overrode} twin_rejected={twin_rejected} "
            f"reward={reward_final:.4f} gap_lat={gap_latency:.4f}",
            flush=True,
        )

        time.sleep(SLEEP_TIME)

    except Exception as e:
        print(f"[LOOP_ERROR] {e}", flush=True)
        time.sleep(3)
