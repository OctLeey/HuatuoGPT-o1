"""Evaluate Clinical-RLVR prediction JSON files offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinical_rlvr.evaluation import evaluate_records, load_prediction_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Clinical-RLVR model outputs.")
    parser.add_argument("--input", required=True, help="Prediction JSON file with `output` or `model_output` fields.")
    parser.add_argument("--output", help="Optional path for the evaluation report JSON.")
    parser.add_argument("--source", help="Override source name for all records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_prediction_records(args.input)
    report = evaluate_records(records, source=args.source)

    printable = {key: value for key, value in report.items() if key != "details"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
