"""Generate model outputs for Clinical-RLVR RLVR records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinical_rlvr.generation import build_mock_predictions, load_rlvr_records, save_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Clinical-RLVR outputs.")
    parser.add_argument("--input", required=True, help="RLVR JSON file.")
    parser.add_argument("--output", required=True, help="Prediction JSON output path.")
    parser.add_argument("--limit", type=int, help="Optional number of records to generate.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock outputs for local testing.")
    parser.add_argument("--mock_mode", choices=["correct", "wrong"], default="correct")
    parser.add_argument("--model_name_or_path", help="Base model path for real generation.")
    parser.add_argument("--adapter_path", help="Optional PEFT adapter path.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def generate_with_model(args: argparse.Namespace, records: list[dict]) -> list[dict]:
    """Generate outputs with a local Hugging Face causal LM."""

    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required unless --mock is used")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    predictions = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        prompts = []
        for record in batch:
            prompt = record["prompt"]
            if getattr(tokenizer, "chat_template", None):
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            prompts.append(prompt)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        for record, text in zip(batch, decoded):
            item = dict(record)
            item["model_output"] = text.strip()
            item["generation_mode"] = "model"
            predictions.append(item)
    return predictions


def main() -> None:
    args = parse_args()
    records = load_rlvr_records(args.input, limit=args.limit)
    if args.mock:
        predictions = build_mock_predictions(records, mode=args.mock_mode)
    else:
        predictions = generate_with_model(args, records)
    save_predictions(args.output, predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
