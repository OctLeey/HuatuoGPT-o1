"""Generation helpers for Clinical-RLVR prediction files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_rlvr_records(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load RLVR prompt records."""

    with Path(path).open(encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("RLVR input must be a JSON list")
    if limit is not None:
        records = records[:limit]
    return records


def build_mock_output(record: dict[str, Any], mode: str = "correct") -> str:
    """Build deterministic mock outputs for local pipeline tests."""

    answer_key = str(record.get("answer_key", "")).upper()
    options = record.get("options", {})
    if mode == "wrong":
        for key in sorted(options):
            if key != answer_key:
                answer_key = key
                break
    elif mode != "correct":
        raise ValueError("mock mode must be 'correct' or 'wrong'")

    answer_text = options.get(answer_key, record.get("answer", ""))
    rationale = record.get("rationale") or f"The best supported option is {answer_key}: {answer_text}."
    return f"<think>\n{rationale}\n</think>\n<answer>\n{answer_key}\n</answer>"


def build_mock_predictions(records: list[dict[str, Any]], mode: str = "correct") -> list[dict[str, Any]]:
    """Attach deterministic mock outputs to RLVR records."""

    predictions = []
    for record in records:
        item = dict(record)
        item["model_output"] = build_mock_output(record, mode=mode)
        item["generation_mode"] = f"mock_{mode}"
        predictions.append(item)
    return predictions


def save_predictions(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Save prediction records as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
