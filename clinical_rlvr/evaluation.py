"""Offline evaluation for Clinical-RLVR outputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from .data import normalize_sample
from .rewards import (
    answer_reward,
    composite_train_reward,
    entity_reward,
    extract_final_answer,
    format_reward,
    length_penalty,
)


def get_model_output(record: dict[str, Any]) -> str:
    """Return the model output from a prediction record."""

    output = record.get("model_output", record.get("output", ""))
    return output if isinstance(output, str) else ""


def reasoning_token_count(output: str) -> int:
    """Count whitespace tokens in the reasoning section when available."""

    if not isinstance(output, str) or not output.strip():
        return 0
    lower = output.lower()
    start_tag = "<think>"
    end_tag = "</think>"
    if start_tag in lower and end_tag in lower:
        start = lower.index(start_tag) + len(start_tag)
        end = lower.index(end_tag)
        return len(output[start:end].split())
    return len(output.split())


def classify_error(record: dict[str, Any], output: str) -> str:
    """Assign a coarse error type for analysis."""

    pred = extract_final_answer(output, record.get("options"))
    answer_key = str(record.get("answer_key", "")).upper()
    fmt = format_reward(output)
    ent = entity_reward(output, record)
    length = length_penalty(output)

    if pred is None:
        return "invalid_output_format"
    if pred != answer_key:
        if fmt >= 1.0 and ent > 0.0:
            return "reasoning_plausible_but_final_option_wrong"
        return "final_answer_wrong"
    if fmt < 1.0:
        return "answer_correct_but_format_invalid"
    if length >= 1.0:
        return "overly_long_or_repetitive_reasoning"
    if ent == 0.0:
        return "answer_correct_but_medical_reasoning_weak"
    return "correct"


def evaluate_record(raw_record: dict[str, Any], source: str | None = None, idx: int | None = None) -> dict[str, Any]:
    """Evaluate one prediction record."""

    record = normalize_sample(raw_record, source=source or raw_record.get("source"), idx=idx)
    output = get_model_output(raw_record)
    reward = composite_train_reward(output, record)
    pred = extract_final_answer(output, record.get("options"))
    result = {
        "id": record["id"],
        "source": record["source"],
        "answer_key": record["answer_key"],
        "pred_answer": pred,
        "is_correct": answer_reward(output, record) == 1.0,
        "format_score": format_reward(output),
        "entity_f1": entity_reward(output, record),
        "length_penalty": length_penalty(output),
        "reasoning_tokens": reasoning_token_count(output),
        "reward": reward["total"],
        "error_type": classify_error(record, output),
    }
    if "judge_score" in raw_record:
        result["judge_score"] = float(raw_record["judge_score"])
    if "judge_label" in raw_record:
        result["judge_label"] = raw_record["judge_label"]
    return result


def summarize_error_types(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count error types in evaluated records."""

    return dict(Counter(item["error_type"] for item in records))


def evaluate_records(raw_records: list[dict[str, Any]], source: str | None = None) -> dict[str, Any]:
    """Evaluate prediction records and return aggregate metrics plus details."""

    details = [evaluate_record(item, source=source, idx=idx) for idx, item in enumerate(raw_records)]
    if not details:
        return {
            "num_samples": 0,
            "accuracy": 0.0,
            "format_rate": 0.0,
            "judge_pass_rate": None,
            "avg_entity_f1": 0.0,
            "avg_reasoning_tokens": 0.0,
            "avg_reward": 0.0,
            "error_types": {},
            "judge_labels": {},
            "details": [],
        }

    judged = [item for item in details if "judge_score" in item]
    return {
        "num_samples": len(details),
        "accuracy": mean(1.0 if item["is_correct"] else 0.0 for item in details),
        "format_rate": mean(1.0 if item["format_score"] >= 1.0 else 0.0 for item in details),
        "judge_pass_rate": mean(item["judge_score"] for item in judged) if judged else None,
        "avg_entity_f1": mean(item["entity_f1"] for item in details),
        "avg_reasoning_tokens": mean(item["reasoning_tokens"] for item in details),
        "avg_reward": mean(item["reward"] for item in details),
        "error_types": summarize_error_types(details),
        "judge_labels": dict(Counter(item["judge_label"] for item in details if "judge_label" in item)),
        "details": details,
    }


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    """Load prediction records from a list or dict-of-lists JSON file."""

    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        records = []
        for source, items in data.items():
            if not isinstance(items, list):
                continue
            for item in items:
                records.append(dict(item, source=item.get("source", source)))
        return records
    raise ValueError("prediction JSON must be a list or a dict of lists")
