import json
from typing import Optional

# Action registry 
ACTION_MAP = {
    0: "no_action",
    1: "block_suspicious_flow",
    2: "limit_bandwidth",
    3: "redirect_traffic",
    4: "isolate_device",
}

VALID_ACTIONS = list(ACTION_MAP.values())
VALID_ACTIONS_STR = ", ".join(VALID_ACTIONS)

# State field labels 
STATE_LABELS = [
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


def _format_state(state_dict: dict) -> str:
    lines = []
    for k, v in state_dict.items():
        lines.append(f"  - {k}: {v}")
    return "\n".join(lines)


def _format_qos(qos: dict) -> str:
    return (
        f"  - Latency  : {qos.get('latency', 'N/A')} ms\n"
        f"  - Packet loss: {qos.get('packet_loss', 'N/A')} %\n"
        f"  - Throughput : {qos.get('throughput', 'N/A')} Mbps"
    )


# Template 1: Explanation
def build_explanation_prompt(
    state_dict: dict,
    action_id: int,
    qos: dict,
    attack_context: Optional[str] = None
) -> str:
    action_name = ACTION_MAP.get(action_id, "unknown")
    context_line = f"Detected attack type: {attack_context}" if attack_context else "No attack type confirmed yet."

    return f"""You are analyzing a defensive decision made by an RL agent in a live SDN network.

=== NETWORK STATE (measured values — use ONLY these) ===
{_format_state(state_dict)}

=== RL AGENT DECISION ===
Action selected: {action_name}  (id={action_id})
{context_line}

=== QoS AFTER ACTION ===
{_format_qos(qos)}

TASKS — answer each section using ONLY the data above:
1. WHY: Why did the RL agent likely choose "{action_name}" given this state?
2. QoS IMPACT: How has the action affected latency, packet loss, and throughput?
3. SAFETY VERDICT: Is the action SAFE or RISKY? Justify in one sentence.
4. SUMMARY: Provide a 2–3 sentence operator-facing explanation.

Output format (plain text, no markdown):
WHY: ...
QoS IMPACT: ...
SAFETY VERDICT: SAFE | RISKY — [one sentence]
SUMMARY: ...
"""


# Template 2: Incident Report 
def build_incident_report_prompt(
    state_dict: dict,
    action_id: int,
    qos_before: dict,
    qos_after: dict,
    attack_type: str = "unknown",
    timestamp: str = "N/A"
) -> str:
    action_name = ACTION_MAP.get(action_id, "unknown")

    return f"""Generate a concise network security incident report.

=== INPUT DATA (ONLY use these values — never invent numbers) ===
Timestamp        : {timestamp}
Attack type      : {attack_type}
RL action taken  : {action_name} (id={action_id})

Network state at time of action:
{_format_state(state_dict)}

QoS BEFORE action:
{_format_qos(qos_before)}

QoS AFTER action:
{_format_qos(qos_after)}

REPORT TEMPLATE — fill in each field using ONLY the data above:
INCIDENT SUMMARY  : [1 sentence: what happened]
ATTACK DETAILS    : [type, observed indicators from state]
ACTION TAKEN      : [what the RL agent did and why it was appropriate]
QoS IMPACT        : [compare before vs after using the numbers provided]
SEVERITY          : [LOW / MEDIUM / HIGH — justify with one fact from the data]
RECOMMENDATION    : [1–2 sentences for the network operator]
"""


# Template 3: Security Query
def build_security_query_prompt(
    question: str,
    network_context: dict
) -> str:
    ctx_str = json.dumps(network_context, indent=2, ensure_ascii=False)

    return f"""A network administrator asks the following question about the current SDN security status.

=== CURRENT NETWORK CONTEXT (ground truth — do NOT fabricate any values) ===
{ctx_str}

=== ADMINISTRATOR QUESTION ===
{question}

Answer the question concisely and factually, referencing only the data provided above.
If the answer cannot be determined from the context, say: "Insufficient data to answer."
Do NOT speculate or use general knowledge to fill gaps in the data.
"""


# Template 4: Faithfulness Probe 
def build_faithfulness_probe_prompt(
    llm_response: str,
    source_data: dict
) -> str:
    src_str = json.dumps(source_data, indent=2, ensure_ascii=False)

    return f"""You are evaluating whether an LLM-generated security report is faithful to its source data.

=== SOURCE DATA PROVIDED TO THE LLM ===
{src_str}

=== LLM-GENERATED RESPONSE ===
{llm_response}

EVALUATION TASK:
1. List any numerical values or claims in the response that CONTRADICT or are ABSENT from the source data. 
   Label each as HALLUCINATION.
2. List values that correctly reflect the source data.
   Label each as FAITHFUL.
3. Give a final Faithfulness Score: fraction of claims that are FAITHFUL (e.g., 8/10 = 0.8).

Output ONLY valid JSON (no markdown):
{{
  "hallucinations": ["<claim>", ...],
  "faithful_claims": ["<claim>", ...],
  "faithfulness_score": <float 0.0–1.0>,
  "verdict": "PASS" | "FAIL"
}}
"""


# Template 5: LLM-as-Judge (qualitative)
def build_judge_prompt(
    llm_response: str,
    action_taken: str,
    state_dict: dict
) -> str:
    state_str = json.dumps(state_dict, ensure_ascii=False)

    return f"""You are an expert evaluator scoring an AI-generated SDN security report.

=== NETWORK STATE ===
{state_str}

=== ACTION TAKEN ===
{action_taken}

=== REPORT TO EVALUATE ===
{llm_response}

Score the report on 3 criteria, each from 1 (poor) to 5 (excellent):
- Accuracy   : Does the report correctly describe the action and its implications given the state?
- Clarity    : Is the report clear and understandable to a network operator?
- Usefulness : Does the report provide actionable insight for the operator?

Output ONLY valid JSON (no markdown):
{{
  "accuracy": <1-5>,
  "clarity": <1-5>,
  "usefulness": <1-5>,
  "justification": "<one sentence>"
}}
"""