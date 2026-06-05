"""Build Clinical-RLVR SFT, DPO, and RLVR datasets from medical QA JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinical_rlvr.datasets import build_datasets, load_raw_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Clinical-RLVR dataset files.")
    parser.add_argument("--input", required=True, help="Input JSON file.")
    parser.add_argument("--output_dir", required=True, help="Directory for generated JSON files.")
    parser.add_argument("--source", help="Override source name for all records.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of input samples.")
    return parser.parse_args()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    raw_samples = load_raw_samples(args.input, source=args.source)
    datasets = build_datasets(raw_samples, source=args.source, limit=args.limit)
    output_dir = Path(args.output_dir)

    write_json(output_dir / "normalized.json", datasets["normalized"])
    write_json(output_dir / "sft.json", datasets["sft"])
    write_json(output_dir / "dpo.json", datasets["dpo"])
    write_json(output_dir / "rlvr.json", datasets["rlvr"])

    summary = {
        "input": args.input,
        "output_dir": str(output_dir),
        "num_samples": len(datasets["normalized"]),
        "files": ["normalized.json", "sft.json", "dpo.json", "rlvr.json"],
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
