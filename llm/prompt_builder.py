import json
from typing import Optional


ACTION_MAP = {
    0: "no_action",
    1: "block_suspicious_flow",
    2: "limit_bandwidth",
    3: "redirect_traffic",
    4: "isolate_device",
}


VALID_ACTIONS = list(ACTION_MAP.values())
VALID_ACTIONS_STR = ", ".join(VALID_ACTIONS)


STATE_LABELS = [
    "packet_rate (pkt/s)",
    "byte_rate (Bps)",
    "flow_count",
    "flow_growth_rate",
    "src_ip_entropy (bits)",
    "latency (ms)",
    "packet_loss (%)",
    "controller_cpu (0-100)",
]


def _format_state(state_dict: dict) -> str:
    lines = []

    for key, value in state_dict.items():
        lines.append(f"  - {key}: {value}")

    return "\n".join(lines)


def _format_qos(qos: dict) -> str:
    return (
        f"  - Latency    : {qos.get('latency', 'N/A')} ms\n"
        f"  - Packet loss: {qos.get('packet_loss', 'N/A')} %\n"
        f"  - Throughput : {qos.get('throughput', 'N/A')}"
    )


def build_explanation_prompt(
    state_dict: dict,
    action_id: int,
    qos: dict,
    attack_context: Optional[str] = None,
) -> str:
    action_name = ACTION_MAP.get(action_id, "unknown")

    context_line = (
        f"Detected attack context: {attack_context}"
        if attack_context
        else "No attack type confirmed yet."
    )

    return f"""You are analyzing a defensive decision made by an RL agent in a live SDN network.

STRICT GROUNDING RULES:
- Use ONLY the values provided below.
- Do NOT invent missing metrics.
- If a metric is missing or N/A, explicitly write DATA UNAVAILABLE.
- Do NOT claim that an attack type is confirmed unless it appears in the input.

=== NETWORK STATE, 8 FEATURES ===
{_format_state(state_dict)}

=== RL AGENT DECISION ===
Action selected: {action_name} (id={action_id})
{context_line}

=== QoS AFTER ACTION ===
{_format_qos(qos)}

TASKS:
1. WHY: Why did the RL agent likely choose "{action_name}" given this state?
2. QoS IMPACT: How has the action affected latency, packet loss, and throughput?
3. SAFETY VERDICT: Is the action SAFE or RISKY? Justify in one sentence.
4. SUMMARY: Provide a 2–3 sentence operator-facing explanation.

Output format:
WHY: ...
QoS IMPACT: ...
SAFETY VERDICT: SAFE | RISKY — ...
SUMMARY: ...
"""


def build_incident_report_prompt(
    state_dict: dict,
    action_id: int,
    qos_before: dict,
    qos_after: dict,
    attack_type: str = "unknown",
    timestamp: str = "N/A",
) -> str:
    action_name = ACTION_MAP.get(action_id, "unknown")

    return f"""Generate a concise network security incident report.

STRICT GROUNDING RULES:
- Only use values in the input.
- Never invent missing values.
- If the evidence is insufficient, say "Insufficient evidence".

=== INPUT DATA ===
Timestamp       : {timestamp}
Attack type     : {attack_type}
RL action taken : {action_name} (id={action_id})

Network state at time of action:
{_format_state(state_dict)}

QoS BEFORE action:
{_format_qos(qos_before)}

QoS AFTER action:
{_format_qos(qos_after)}

REPORT TEMPLATE:
INCIDENT SUMMARY : ...
ATTACK DETAILS   : ...
ACTION TAKEN     : ...
QoS IMPACT       : ...
SEVERITY         : LOW | MEDIUM | HIGH — ...
RECOMMENDATION   : ...
"""


def build_security_query_prompt(
    question: str,
    network_context: dict,
) -> str:
    context = json.dumps(network_context, indent=2, ensure_ascii=False)

    return f"""A network administrator asks the following question about the current SDN security status.

=== CURRENT NETWORK CONTEXT ===
{context}

=== ADMINISTRATOR QUESTION ===
{question}

Answer concisely and factually using only the context.
If the answer cannot be determined from the context, say:
"Insufficient data to answer."
"""


def build_faithfulness_probe_prompt(
    llm_response: str,
    source_data: dict,
) -> str:
    source = json.dumps(source_data, indent=2, ensure_ascii=False)

    return f"""You are evaluating whether an LLM-generated SDN security report is faithful to its source data.

=== SOURCE DATA PROVIDED TO THE LLM ===
{source}

=== LLM-GENERATED RESPONSE ===
{llm_response}

EVALUATION TASK:
1. List numerical values or claims in the response that contradict or are absent from the source data.
2. List values that correctly reflect the source data.
3. Give a final Faithfulness Score from 0.0 to 1.0.

Output ONLY valid JSON:
{{
  "hallucinations": ["<claim>", "..."],
  "faithful_claims": ["<claim>", "..."],
  "faithfulness_score": <float>,
  "verdict": "PASS" | "FAIL"
}}
"""


def build_judge_prompt(
    llm_response: str,
    action_taken: str,
    state_dict: dict,
) -> str:
    state = json.dumps(state_dict, ensure_ascii=False)

    return f"""You are an expert evaluator scoring an AI-generated SDN security report.

=== NETWORK STATE ===
{state}

=== ACTION TAKEN ===
{action_taken}

=== REPORT TO EVALUATE ===
{llm_response}

Score the report on 3 criteria, each from 1 to 5:
- Accuracy: Does the report correctly describe the action and implications?
- Clarity: Is it clear for a network operator?
- Usefulness: Does it provide actionable insight?

Output ONLY valid JSON:
{{
  "accuracy": <1-5>,
  "clarity": <1-5>,
  "usefulness": <1-5>,
  "justification": "<one sentence>"
}}
"""