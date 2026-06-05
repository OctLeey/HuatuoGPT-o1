"""Attach LLM-as-a-Judge labels to Clinical-RLVR prediction files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinical_rlvr.evaluation import load_prediction_records
from clinical_rlvr.judge import (
    attach_judge_result,
    cache_key,
    call_openai_compatible_judge,
    load_judge_cache,
    mock_judge,
    save_judge_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge Clinical-RLVR predictions.")
    parser.add_argument("--input", required=True, help="Prediction JSON file.")
    parser.add_argument("--output", required=True, help="Judged prediction JSON output path.")
    parser.add_argument("--cache", help="Optional JSON cache path.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic local mock judge.")
    parser.add_argument("--model_name", default="gpt-4o-mini", help="Judge model name for API mode.")
    parser.add_argument("--api_url", default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--limit", type=int, help="Optional maximum number of predictions to judge.")
    return parser.parse_args()


def write_json(path: str | Path, data: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    records = load_prediction_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    if not args.mock and not args.api_key:
        raise ValueError("--api_key or OPENAI_API_KEY is required unless --mock is used")

    cache = load_judge_cache(args.cache)
    judged_records = []
    for record in records:
        key = cache_key(record)
        if key in cache:
            result = dict(cache[key])
            result["judge_cache_hit"] = True
        else:
            if args.mock:
                result = mock_judge(record)
            else:
                result = call_openai_compatible_judge(
                    record=record,
                    model_name=args.model_name,
                    api_url=args.api_url,
                    api_key=args.api_key,
                )
            result["judge_cache_hit"] = False
            cache[key] = result
        judged_records.append(attach_judge_result(record, result))

    write_json(args.output, judged_records)
    save_judge_cache(args.cache, cache)
    print(f"Wrote {len(judged_records)} judged predictions to {args.output}")


if __name__ == "__main__":
    main()
