"""Dataset builders for Clinical-RLVR.

The builders produce three small, explicit formats:

- SFT: `messages`, `prompt`, and `response`.
- DPO: `prompt`, `chosen`, and `rejected`.
- RLVR: prompt plus reference fields for reward computation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import format_mcq_prompt, format_sft_response, normalize_sample


def load_raw_samples(path: str | Path, source: str | None = None) -> list[dict[str, Any]]:
    """Load a HuatuoGPT-o1 style JSON file into raw samples.

    The input can be a plain list or a dict of split/source names to lists, which
    matches the structure used by `evaluation/data/eval_data.json`.
    """

    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [dict(item, source=source or item.get("source", "unknown")) for item in data]
    if isinstance(data, dict):
        samples = []
        for group_name, group_items in data.items():
            if not isinstance(group_items, list):
                continue
            for item in group_items:
                samples.append(dict(item, source=source or group_name))
        return samples
    raise ValueError("input JSON must be a list or a dict of lists")


def build_sft_record(sample: dict[str, Any]) -> dict[str, Any]:
    """Build one SFT training record."""

    prompt = format_mcq_prompt(sample, strict=True)
    response = format_sft_response(sample.get("rationale", ""), sample["answer_key"])
    return {
        "id": sample["id"],
        "source": sample["source"],
        "prompt": prompt,
        "response": response,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "answer_key": sample["answer_key"],
        "answer": sample["answer"],
    }


def _choose_wrong_answer_key(sample: dict[str, Any]) -> str:
    for key in sorted(sample["options"]):
        if key != sample["answer_key"]:
            return key
    raise ValueError("sample must contain at least one wrong option")


def build_dpo_record(sample: dict[str, Any]) -> dict[str, Any]:
    """Build one DPO preference record with a deterministic hard negative."""

    rejected_key = _choose_wrong_answer_key(sample)
    rejected_option = sample["options"][rejected_key]
    rejected_rationale = (
        "This response follows the requested format but selects an incorrect "
        f"medical option: {rejected_option}."
    )
    return {
        "id": sample["id"],
        "source": sample["source"],
        "prompt": format_mcq_prompt(sample, strict=True),
        "chosen": format_sft_response(sample.get("rationale", ""), sample["answer_key"]),
        "rejected": format_sft_response(rejected_rationale, rejected_key),
        "chosen_answer_key": sample["answer_key"],
        "rejected_answer_key": rejected_key,
        "answer": sample["answer"],
    }


def build_rlvr_record(sample: dict[str, Any]) -> dict[str, Any]:
    """Build one RLVR prompt record with references for reward computation."""

    return {
        "id": sample["id"],
        "source": sample["source"],
        "prompt": format_mcq_prompt(sample, strict=True),
        "question": sample["question"],
        "options": sample["options"],
        "answer_key": sample["answer_key"],
        "answer": sample["answer"],
        "rationale": sample.get("rationale", ""),
        "verifiable_answer": sample["verifiable_answer"],
        "task_type": sample["task_type"],
    }


def build_datasets(raw_samples: list[dict[str, Any]], source: str | None = None, limit: int | None = None) -> dict[str, list[dict[str, Any]]]:
    """Normalize raw samples and build SFT, DPO, and RLVR datasets."""

    selected = raw_samples[:limit] if limit is not None else raw_samples
    normalized = [normalize_sample(item, source=source or item.get("source"), idx=idx) for idx, item in enumerate(selected)]
    return {
        "normalized": normalized,
        "sft": [build_sft_record(item) for item in normalized],
        "dpo": [build_dpo_record(item) for item in normalized],
        "rlvr": [build_rlvr_record(item) for item in normalized],
    }
