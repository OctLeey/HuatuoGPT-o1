"""LLM-as-a-Judge verifier helpers for Clinical-RLVR."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .evaluation import get_model_output
from .rewards import answer_reward


VALID_LABELS = {"correct", "incorrect", "uncertain"}


def build_judge_prompt(record: dict[str, Any]) -> str:
    """Build a concise medical semantic equivalence judge prompt."""

    question = record.get("question", "")
    options = record.get("options", {})
    option_text = "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))
    reference = record.get("verifiable_answer") or record.get("answer", "")
    output = get_model_output(record)
    return f"""You are a strict medical QA evaluator.

Decide whether the model answer is medically equivalent to the reference answer.
Focus on the final medical answer, not writing style. If the answer is partially correct but misses the key diagnosis, treatment, drug, test, or option, mark it as incorrect. If the evidence is insufficient, mark it as uncertain.

Question:
{question}

Options:
{option_text}

Reference Answer:
{reference}

Model Answer:
{output}

Return exactly one label: correct, incorrect, or uncertain."""


def parse_judge_label(text: str) -> str:
    """Parse a judge response into a stable label."""

    lowered = (text or "").strip().lower()
    for label in ["correct", "incorrect", "uncertain"]:
        if lowered == label or lowered.startswith(label):
            return label
    if "incorrect" in lowered:
        return "incorrect"
    if "uncertain" in lowered or "not sure" in lowered:
        return "uncertain"
    if "correct" in lowered:
        return "correct"
    return "uncertain"


def judge_score(label: str) -> float:
    """Map judge label to a numeric score."""

    if label == "correct":
        return 1.0
    if label == "uncertain":
        return 0.3
    return 0.0


def cache_key(record: dict[str, Any]) -> str:
    """Return a stable cache key for one judged prediction."""

    payload = {
        "id": record.get("id"),
        "question": record.get("question"),
        "answer": record.get("answer"),
        "verifiable_answer": record.get("verifiable_answer"),
        "output": get_model_output(record),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_judge_cache(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load judge cache if present."""

    if path is None:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    with cache_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("judge cache must be a JSON object")
    return data


def save_judge_cache(path: str | Path | None, cache: dict[str, dict[str, Any]]) -> None:
    """Save judge cache."""

    if path is None:
        return
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def mock_judge(record: dict[str, Any]) -> dict[str, Any]:
    """Deterministic local judge used for tests and pipeline dry-runs."""

    label = "correct" if answer_reward(get_model_output(record), record) == 1.0 else "incorrect"
    return {
        "judge_label": label,
        "judge_score": judge_score(label),
        "judge_response": label,
        "judge_mode": "mock",
    }


def call_openai_compatible_judge(
    record: dict[str, Any],
    model_name: str,
    api_url: str,
    api_key: str,
    timeout: int = 60,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat completions endpoint."""

    import requests

    prompt = build_judge_prompt(record)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 16,
    }
    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    label = parse_judge_label(content)
    return {
        "judge_label": label,
        "judge_score": judge_score(label),
        "judge_response": content,
        "judge_mode": "api",
    }


def attach_judge_result(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a prediction record with judge fields attached."""

    judged = dict(record)
    judged.update(result)
    return judged
