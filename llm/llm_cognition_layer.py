import csv
import os
import time
from typing import Any, Dict, Optional

import numpy as np

from llm.llm_service import call_llm
from llm.prompt_builder import build_explanation_prompt


ACTION_MAP = {
    0: "no_action",
    1: "block_suspicious_flow",
    2: "limit_bandwidth",
    3: "redirect_traffic",
    4: "isolate_device",
}


def state_vector_to_dict(state) -> Dict[str, float]:
    arr = np.asarray(state, dtype=float).reshape(-1)

    if len(arr) < 8:
        raise ValueError(f"Expected 8-dim state, got {len(arr)}")

    return {
        "packet_rate": round(float(arr[0]), 6),
        "byte_rate": round(float(arr[1]), 6),
        "flow_count": round(float(arr[2]), 6),
        "flow_growth_rate": round(float(arr[3]), 6),
        "src_ip_entropy": round(float(arr[4]), 6),
        "latency": round(float(arr[5]), 6),
        "packet_loss": round(float(arr[6]), 6),
        "controller_cpu": round(float(arr[7]), 6),
    }


def explain_decision(
    state_dict: Dict[str, Any],
    action_id: int,
    qos: Dict[str, Any],
    attack_context: Optional[str] = None,
) -> str:
    prompt = build_explanation_prompt(
        state_dict=state_dict,
        action_id=int(action_id),
        qos=qos,
        attack_context=attack_context,
    )

    result = call_llm(
        prompt=prompt,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "350")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    )

    return str(result.get("text", ""))


def log_decision(
    state_dict: Dict[str, Any],
    action_id: int,
    qos: Dict[str, Any],
    explanation: str,
    log_path: Optional[str] = None,
) -> None:
    if log_path is None:
        log_path = os.getenv("LLM_DECISION_LOG", "logs/llm_decisions.csv")

    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    header = [
        "ts",
        "action_id",
        "action_name",
        "packet_rate",
        "byte_rate",
        "flow_count",
        "flow_growth_rate",
        "src_ip_entropy",
        "latency",
        "packet_loss",
        "controller_cpu",
        "qos_latency",
        "qos_packet_loss",
        "qos_throughput",
        "explanation",
    ]

    file_exists = os.path.exists(log_path)

    with open(log_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(header)

        writer.writerow(
            [
                time.time(),
                int(action_id),
                ACTION_MAP.get(int(action_id), "unknown"),
                state_dict.get("packet_rate", ""),
                state_dict.get("byte_rate", ""),
                state_dict.get("flow_count", ""),
                state_dict.get("flow_growth_rate", ""),
                state_dict.get("src_ip_entropy", ""),
                state_dict.get("latency", ""),
                state_dict.get("packet_loss", ""),
                state_dict.get("controller_cpu", ""),
                qos.get("latency", ""),
                qos.get("packet_loss", ""),
                qos.get("throughput", ""),
                str(explanation).replace("\n", " ")[:2000],
            ]
        )