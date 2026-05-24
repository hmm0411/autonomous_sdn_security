import os
import sys
import json
from datetime import datetime
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.llm_service import call_llm
from llm.prompt_builder import (
    build_explanation_prompt,
    build_incident_report_prompt,
    build_security_query_prompt,
    ACTION_MAP,
)

# ─── Log file ─────────────────────────────────────────────────────────────────
DEFAULT_LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "llm_reports.log")


def _safe_call(prompt: str, max_tokens: int = 350) -> str:
    """Wrapper – always returns a string, never raises."""
    try:
        result = call_llm(prompt, max_tokens=max_tokens)
        return result["text"]
    except Exception as e:
        return f"[LLM UNAVAILABLE] {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def explain_decision(
    state_dict: dict,
    action_id: int,
    qos: dict,
    attack_context: Optional[str] = None,
) -> str:
    """
    Generate a natural-language explanation for a single RL decision.

    Parameters
    ----------
    state_dict   : dict  – named network features (e.g. from StateBuilder)
    action_id    : int   – RL action (0–4)
    qos          : dict  – {'latency': ms, 'packet_loss': %, 'throughput': Mbps}
    attack_context : str – optional attack type string

    Returns
    -------
    str – operator-facing explanation (4 sections: WHY / QoS IMPACT / SAFETY / SUMMARY)
    """
    prompt = build_explanation_prompt(state_dict, action_id, qos, attack_context)
    return _safe_call(prompt)


def generate_incident_report(
    state_dict: dict,
    action_id: int,
    qos_before: dict,
    qos_after: dict,
    attack_type: str = "unknown",
    timestamp: Optional[str] = None,
) -> str:
    """
    Generate a structured incident report after a defense action.

    Parameters
    ----------
    state_dict   : dict  – network state at time of action
    action_id    : int   – RL action taken
    qos_before   : dict  – QoS metrics before action
    qos_after    : dict  – QoS metrics after action
    attack_type  : str   – label from IDS (e.g. "DDoS flood")
    timestamp    : str   – ISO timestamp; auto-generated if None

    Returns
    -------
    str – formatted incident report
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prompt = build_incident_report_prompt(
        state_dict, action_id, qos_before, qos_after, attack_type, timestamp
    )
    return _safe_call(prompt, max_tokens=400)


def query_network_status(
    question: str,
    network_context: dict,
) -> str:
    """
    Answer an operator's natural-language question about network security status.

    Parameters
    ----------
    question        : str  – admin's question
    network_context : dict – current state, recent actions, QoS, etc.

    Returns
    -------
    str – factual answer grounded in network_context
    """
    prompt = build_security_query_prompt(question, network_context)
    return _safe_call(prompt, max_tokens=300)


def log_decision(
    state_dict: dict,
    action_id: int,
    qos: dict,
    explanation: str,
    log_path: Optional[str] = None,
) -> None:
    """
    Append a decision log entry to the reports log file.

    Entry format:
      [TIMESTAMP] ACTION=<name> | <first 200 chars of explanation>
    """
    path = log_path or DEFAULT_LOG
    os.makedirs(os.path.dirname(path), exist_ok=True)

    action_name = ACTION_MAP.get(action_id, f"action_{action_id}")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snippet = explanation.replace("\n", " ")[:200]

    line = f"[{ts}] ACTION={action_name} | {snippet}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: build state_dict from raw numpy/list state vector
# ═══════════════════════════════════════════════════════════════════════════════

STATE_KEYS = [
    "packet_rate (pkt/s)",
    "byte_rate (Bps)",
    "flow_count",
    "src_ip_entropy (bits)",
    "latency (ms)",
    "packet_loss (%)",
    "queue_length",
    "controller_cpu (0-1)",
    "previous_action",
]


def state_vector_to_dict(state_vector) -> dict:
    """Convert a 9-element state array to a labelled dict for LLM prompts."""
    vec = list(state_vector)
    return {k: round(float(v), 4) for k, v in zip(STATE_KEYS, vec)}


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== LLM Cognition Layer — Demo ===\n")

    demo_state = {
        "packet_rate (pkt/s)":   4850.0,
        "byte_rate (Bps)":       2350000.0,
        "flow_count":            342,
        "src_ip_entropy (bits)": 0.3,
        "latency (ms)":          187.6,
        "packet_loss (%)":       0.38,
        "queue_length":          210.0,
        "controller_cpu (0-1)":  0.91,
        "previous_action":       0,
    }
    demo_qos = {"latency": 187.6, "packet_loss": 0.38, "throughput": 12.1}

    print("--- explain_decision ---")
    exp = explain_decision(demo_state, action_id=1, qos=demo_qos, attack_context="DDoS flood")
    print(exp)

    print("\n--- generate_incident_report ---")
    qos_after = {"latency": 45.0, "packet_loss": 0.02, "throughput": 85.0}
    report = generate_incident_report(demo_state, action_id=1, qos_before=demo_qos, qos_after=qos_after, attack_type="DDoS flood")
    print(report)

    print("\n--- query_network_status ---")
    ctx = {**demo_state, "last_action": "block_suspicious_flow", "alert_count_last_hour": 3}
    ans = query_network_status("Is the network currently under attack?", ctx)
    print(ans)