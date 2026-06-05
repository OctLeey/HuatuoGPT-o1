"""Data helpers for Clinical-RLVR.

The functions in this module normalize HuatuoGPT-o1 style medical QA examples
into a stable schema that can be reused by SFT, DPO, RLVR, and evaluation code.
"""

from __future__ import annotations

from typing import Any


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing required string field: {field_name}")
    return value.strip()


def build_option_text(options: dict[str, str]) -> str:
    """Render multiple-choice options in a deterministic order."""

    if not isinstance(options, dict) or not options:
        raise ValueError("options must be a non-empty dict")
    return "\n".join(f"{key}. {options[key]}" for key in sorted(options))


def normalize_sample(raw: dict[str, Any], source: str | None = None, idx: int | None = None) -> dict[str, Any]:
    """Normalize one medical QA sample to the Clinical-RLVR schema.

    The original repository uses several overlapping field names, such as
    `question`/`Question`, `answer_idx`, and `Ground-True Answer`. This helper
    keeps downstream code from depending on those variants.
    """

    question = raw.get("Question") or raw.get("question")
    answer_key = raw.get("answer_key") or raw.get("answer_idx")
    answer = raw.get("answer") or raw.get("Ground-True Answer") or raw.get("Response")
    rationale = raw.get("rationale") or raw.get("Complex_CoT") or ""
    verifiable_answer = raw.get("verifiable_answer") or raw.get("Ground-True Answer") or answer
    sample_source = source or raw.get("source") or "unknown"
    sample_id = raw.get("id") or raw.get("process_id") or idx

    options = raw.get("options")
    if not isinstance(options, dict) or not options:
        raise ValueError("missing required options field")

    normalized = {
        "id": f"{sample_source}_{sample_id}" if sample_id is not None else f"{sample_source}_unknown",
        "source": sample_source,
        "question": _require_str(question, "question"),
        "options": {str(key).strip(): str(value).strip() for key, value in options.items()},
        "answer_key": _require_str(answer_key, "answer_key").upper(),
        "answer": _require_str(answer, "answer"),
        "rationale": rationale.strip() if isinstance(rationale, str) else "",
        "verifiable_answer": _require_str(verifiable_answer, "verifiable_answer"),
        "task_type": raw.get("task_type") or "multiple_choice_medical_qa",
    }
    if normalized["answer_key"] not in normalized["options"]:
        raise ValueError(f"answer_key {normalized['answer_key']} not present in options")
    return normalized


def format_mcq_prompt(sample: dict[str, Any], strict: bool = False) -> str:
    """Build a prompt for medical multiple-choice QA."""

    question = _require_str(sample.get("question"), "question")
    options = sample.get("options")
    option_text = build_option_text(options)
    if strict:
        prefix = (
            "Please answer the following medical multiple-choice question. "
            "Think through the case, then end with the correct option in "
            "<answer>...</answer>."
        )
    else:
        prefix = "Please answer the following medical multiple-choice question."
    return f"{prefix}\n{question}\n{option_text}"


def format_sft_response(rationale: str, answer_key: str) -> str:
    """Format a response for Qwen3-style CoT SFT."""

    rationale = rationale.strip() or "Reason through the medical evidence and select the best answer."
    answer_key = _require_str(answer_key, "answer_key").upper()
    return f"<think>\n{rationale}\n</think>\n<answer>\n{answer_key}\n</answer>"
