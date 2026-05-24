import os
import time
import json
import re
from huggingface_hub import InferenceClient

# lấy token từ môi trường
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found in environment variables")

client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=HF_TOKEN
)

SYSTEM_PROMPT = """You are an expert SDN (Software-Defined Networking) network security analyst.

STRICT RULES YOU MUST FOLLOW AT ALL TIMES:
1. Only use data explicitly provided in the user's message. NEVER invent, estimate, or extrapolate metrics.
2. If a required metric is missing or marked as N/A, explicitly state "DATA UNAVAILABLE" for that field.
3. When asked to select an action, ONLY choose from the provided action list. Never return any action outside the list.
4. When asked for JSON output, return ONLY valid JSON with no markdown fences, no preamble, no explanation outside the JSON.
5. Keep all explanations concise, factual, and grounded in the provided data.
"""

def call_llm(prompt: str, max_tokens: int = 300, temperature: float = 0.2) -> dict:
    """
    Call the LLM and return a dict with 'text' and 'latency_ms'.
    Always succeeds — errors are returned in 'text' as LLM_ERROR.
    """
    start = time.time()
    try:
        res = client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        text = res.choices[0].message.content
    except Exception as e:
        text = f"LLM_ERROR: {str(e)}"

    latency_ms = (time.time() - start) * 1000
    return {"text": text, "latency_ms": round(latency_ms, 2)}


def call_llm_json(prompt: str, max_tokens: int = 400) -> dict:
    """
    Call LLM expecting a JSON response.
    Returns parsed dict on success; on parse failure returns
    {'parse_error': True, 'raw': <raw text>, 'latency_ms': ...}.
    """
    result = call_llm(prompt, max_tokens=max_tokens, temperature=0.1)
    raw = result["text"]

    # Strip markdown fences if the model ignores the instruction
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        parsed = json.loads(clean)
        parsed["latency_ms"] = result["latency_ms"]
        return parsed
    except json.JSONDecodeError:
        return {
            "parse_error": True,
            "raw": raw,
            "latency_ms": result["latency_ms"]
        }
    