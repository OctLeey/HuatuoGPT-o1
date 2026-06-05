"""Reward helpers for Clinical-RLVR.

These functions are intentionally lightweight and dependency-free. They provide
local, testable reward signals before the project connects them to GRPO/RLVR
training and optional LLM-based judges.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


ANSWER_BLOCK_RE = re.compile(r"<answer>\s*([A-N])\s*</answer>", re.IGNORECASE | re.DOTALL)
FINAL_RESPONSE_RE = re.compile(r"## Final Response\s*(?:\n)+(.+)", re.IGNORECASE | re.DOTALL)
ANSWER_IS_RE = re.compile(r"(?:answer\s+is|answer:)\s*([A-N])\b", re.IGNORECASE)

STOPWORDS = {
    "about",
    "after",
    "answer",
    "based",
    "because",
    "before",
    "being",
    "best",
    "case",
    "cause",
    "clinical",
    "condition",
    "correct",
    "disease",
    "following",
    "given",
    "likely",
    "medical",
    "patient",
    "patients",
    "present",
    "presents",
    "question",
    "response",
    "should",
    "signs",
    "symptoms",
    "therefore",
    "these",
    "this",
    "treatment",
    "which",
    "with",
}


def extract_final_answer(text: str, options: dict[str, str] | None = None) -> str | None:
    """Extract the final option label from a model response."""

    if not isinstance(text, str) or not text.strip():
        return None

    match = ANSWER_BLOCK_RE.search(text)
    if match:
        return match.group(1).upper()

    final_text = text
    match = FINAL_RESPONSE_RE.search(text)
    if match:
        final_text = match.group(1)

    match = ANSWER_IS_RE.search(final_text)
    if match:
        return match.group(1).upper()

    labels = "".join(sorted(options.keys())) if options else "ABCDEFGHIJKLMN"
    matches = re.findall(rf"(?<![A-Za-z])([{re.escape(labels)}])(?![A-Za-z])", final_text.upper())
    if matches:
        return matches[-1].upper()
    return None


def answer_reward(output: str, sample: dict[str, Any]) -> float:
    """Return 1.0 when the extracted final answer matches the reference key."""

    answer_key = str(sample.get("answer_key", "")).upper()
    pred = extract_final_answer(output, sample.get("options"))
    return 1.0 if pred == answer_key else 0.0


def format_reward(output: str) -> float:
    """Reward the expected `<think>` plus `<answer>` output format."""

    if not isinstance(output, str):
        return 0.0
    has_think = bool(re.search(r"<think>.+?</think>", output, re.IGNORECASE | re.DOTALL))
    has_answer = bool(ANSWER_BLOCK_RE.search(output))
    if has_think and has_answer:
        return 1.0
    if extract_final_answer(output) is not None:
        return 0.5
    return 0.0


def _tokenize_entities(text: str) -> set[str]:
    """A lexical entity baseline to be replaced or augmented by cached LLM NER."""

    if not isinstance(text, str):
        return set()
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+-]{2,}", text.lower())
    return {token for token in tokens if token not in STOPWORDS and len(token) >= 4}


def _f1(predicted: Iterable[str], reference: Iterable[str]) -> float:
    pred = set(predicted)
    ref = set(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = len(pred & ref)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def entity_reward(output: str, sample: dict[str, Any]) -> float:
    """Score lexical medical-entity consistency against answer and rationale."""

    reference_text = f"{sample.get('answer', '')}\n{sample.get('rationale', '')}"
    final_text = output
    final_match = ANSWER_BLOCK_RE.search(output or "")
    if final_match and sample.get("options"):
        option = final_match.group(1).upper()
        final_text = f"{output}\n{sample['options'].get(option, option)}"
    return _f1(_tokenize_entities(final_text), _tokenize_entities(reference_text))


def length_penalty(output: str, soft_limit: int = 512, hard_limit: int = 1024) -> float:
    """Return a normalized penalty for overly long reasoning."""

    if not isinstance(output, str) or soft_limit <= 0 or hard_limit <= soft_limit:
        return 0.0
    think_match = re.search(r"<think>(.+?)</think>", output, re.IGNORECASE | re.DOTALL)
    reasoning = think_match.group(1) if think_match else output
    token_count = len(reasoning.split())
    if token_count <= soft_limit:
        return 0.0
    if token_count >= hard_limit:
        return 1.0
    return (token_count - soft_limit) / (hard_limit - soft_limit)


def composite_train_reward(output: str, sample: dict[str, Any]) -> dict[str, float]:
    """Compute the first-version Clinical-RLVR training reward components."""

    components = {
        "answer": answer_reward(output, sample),
        "format": format_reward(output),
        "entity": entity_reward(output, sample),
        "length_penalty": length_penalty(output),
    }
    components["total"] = (
        components["answer"]
        + 0.2 * components["format"]
        + 0.3 * components["entity"]
        - 0.1 * components["length_penalty"]
    )
    return components
