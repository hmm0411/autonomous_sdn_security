import json
import os
import re
import time
from typing import Any, Dict

from huggingface_hub import InferenceClient


HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")


SYSTEM_PROMPT = """You are an expert SDN network security analyst.

STRICT RULES:
1. Only use data explicitly provided by the user.
2. Never invent metrics, attack types, timestamps, or actions.
3. If a metric is missing or marked N/A, explicitly say DATA UNAVAILABLE.
4. If JSON is requested, return only valid JSON.
5. Keep the response concise and grounded.
"""


_client = None


def _get_client():
    global _client

    if _client is not None:
        return _client

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. Export it first: export HF_TOKEN='your_token'"
        )

    _client = InferenceClient(
        model=HF_MODEL,
        token=HF_TOKEN,
    )

    return _client


def call_llm(
    prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Return:
    {
      "text": "...",
      "latency_ms": 123.4
    }
    """
    start = time.time()

    try:
        client = _get_client()

        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text = response.choices[0].message.content

    except Exception as exc:
        text = f"LLM_ERROR: {exc}"

    latency_ms = (time.time() - start) * 1000.0

    return {
        "text": text,
        "latency_ms": round(latency_ms, 2),
    }


def call_llm_json(
    prompt: str,
    max_tokens: int = 400,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    result = call_llm(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    raw_text = result["text"]
    clean_text = re.sub(r"```(?:json)?|```", "", raw_text).strip()

    try:
        parsed = json.loads(clean_text)
        parsed["latency_ms"] = result["latency_ms"]
        return parsed

    except json.JSONDecodeError:
        return {
            "parse_error": True,
            "raw": raw_text,
            "latency_ms": result["latency_ms"],
        }