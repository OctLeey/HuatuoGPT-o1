"""GRPO/RLVR entrypoint for Clinical-RLVR.

The script connects the Clinical-RLVR answer, format, entity, and length reward
signals to Hugging Face TRL's GRPOTrainer. Use `--dry_run` locally to validate
data and reward behavior before running GPU training on AutoDL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinical_rlvr.generation import build_mock_output
from clinical_rlvr.rewards import answer_reward, entity_reward, format_reward, length_penalty
from train.sft_qlora import load_config


REQUIRED_CONFIG_KEYS = {
    "model_name_or_path",
    "train_file",
    "output_dir",
    "max_prompt_length",
    "max_completion_length",
    "num_generations",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "beta",
}


def validate_config(config: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"missing required config keys: {', '.join(missing)}")
    train_file = Path(config["train_file"])
    if not train_file.exists():
        raise FileNotFoundError(f"train_file does not exist: {train_file}")


def load_rlvr_records(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("GRPO train_file must contain a JSON list")
    if limit is not None:
        records = records[:limit]
    for idx, record in enumerate(records):
        for field in ["prompt", "options", "answer_key", "answer"]:
            if field not in record:
                raise ValueError(f"RLVR record {idx} missing required field: {field}")
    return records


def _completion_to_text(completion: Any) -> str:
    """Support TRL standard and conversational completion formats."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return str(completion or "")


def _sample_from_kwargs(index: int, kwargs: dict[str, Any]) -> dict[str, Any]:
    sample = {}
    for key in ["options", "answer_key", "answer", "rationale", "verifiable_answer", "source", "task_type"]:
        values = kwargs.get(key)
        if isinstance(values, list) and index < len(values):
            sample[key] = values[index]
    return sample


def clinical_answer_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    rewards = []
    for idx, completion in enumerate(completions):
        sample = _sample_from_kwargs(idx, kwargs)
        rewards.append(answer_reward(_completion_to_text(completion), sample))
    return rewards


def clinical_format_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    return [format_reward(_completion_to_text(completion)) for completion in completions]


def clinical_entity_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    rewards = []
    for idx, completion in enumerate(completions):
        sample = _sample_from_kwargs(idx, kwargs)
        rewards.append(entity_reward(_completion_to_text(completion), sample))
    return rewards


def clinical_length_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    return [-length_penalty(_completion_to_text(completion)) for completion in completions]


def make_weighted_reward_func(base_func: Any, weight: float, name: str) -> Any:
    def weighted_reward(completions: list[Any], **kwargs: Any) -> list[float]:
        return [weight * value for value in base_func(completions, **kwargs)]

    weighted_reward.__name__ = name
    return weighted_reward


def build_reward_funcs(config: dict[str, Any]) -> list[Any]:
    return [
        make_weighted_reward_func(
            clinical_answer_reward,
            float(config.get("answer_reward_weight", 1.0)),
            "clinical_answer_reward",
        ),
        make_weighted_reward_func(
            clinical_format_reward,
            float(config.get("format_reward_weight", 0.2)),
            "clinical_format_reward",
        ),
        make_weighted_reward_func(
            clinical_entity_reward,
            float(config.get("entity_reward_weight", 0.3)),
            "clinical_entity_reward",
        ),
        make_weighted_reward_func(
            clinical_length_reward,
            abs(float(config.get("length_penalty_weight", -0.1))),
            "clinical_length_penalty",
        ),
    ]


def dry_run(config: dict[str, Any], limit: int = 3) -> dict[str, Any]:
    validate_config(config)
    records = load_rlvr_records(config["train_file"], limit=limit)
    completions = [build_mock_output(record) for record in records]
    kwargs = {
        "options": [record["options"] for record in records],
        "answer_key": [record["answer_key"] for record in records],
        "answer": [record["answer"] for record in records],
        "rationale": [record.get("rationale", "") for record in records],
        "verifiable_answer": [record.get("verifiable_answer", record["answer"]) for record in records],
    }
    reward_funcs = build_reward_funcs(config)
    reward_breakdown = {func.__name__: func(completions, **kwargs) for func in reward_funcs}
    totals = [sum(values) for values in zip(*reward_breakdown.values())] if records else []
    return {
        "model_name_or_path": config["model_name_or_path"],
        "train_file": config["train_file"],
        "checked_records": len(records),
        "output_dir": config["output_dir"],
        "num_generations": config["num_generations"],
        "reward_breakdown": reward_breakdown,
        "reward_totals": totals,
    }


def train(config: dict[str, Any]) -> None:
    validate_config(config)

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import BitsAndBytesConfig, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "GRPO training requires a newer TRL environment with GRPOTrainer. "
            "Install the Clinical-RLVR GRPO requirements on AutoDL."
        ) from exc

    records = load_rlvr_records(config["train_file"])
    dataset = Dataset.from_list(records)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if config.get("bnb_4bit_compute_dtype", "bfloat16") == "bfloat16" else torch.float16
    model_init_kwargs: dict[str, Any] = {
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
    }
    if config.get("use_4bit", True):
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    grpo_args = GRPOConfig(
        output_dir=config["output_dir"],
        run_name=config.get("run_name"),
        max_prompt_length=int(config["max_prompt_length"]),
        max_completion_length=int(config["max_completion_length"]),
        num_generations=int(config["num_generations"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        beta=float(config["beta"]),
        num_train_epochs=float(config.get("num_train_epochs", 1)),
        max_steps=int(config.get("max_steps", -1)),
        temperature=float(config.get("temperature", 0.9)),
        bf16=bool(config.get("bf16", True)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        logging_steps=int(config.get("logging_steps", 5)),
        save_steps=int(config.get("save_steps", 50)),
        save_total_limit=int(config.get("save_total_limit", 2)),
        report_to=config.get("report_to", "none"),
        remove_unused_columns=False,
        model_init_kwargs=model_init_kwargs,
        seed=int(config.get("seed", 42)),
    )

    peft_config = None
    if config.get("use_peft", True):
        peft_config = LoraConfig(
            r=int(config.get("lora_r", 16)),
            lora_alpha=int(config.get("lora_alpha", 32)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            target_modules=list(config.get("lora_target_modules", [])),
            bias="none",
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=config["model_name_or_path"],
        reward_funcs=build_reward_funcs(config),
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Clinical-RLVR GRPO/RLVR training.")
    parser.add_argument("--config", required=True, help="YAML or JSON config file.")
    parser.add_argument("--dry_run", action="store_true", help="Validate config, data, and reward functions.")
    parser.add_argument("--dry_run_limit", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(dry_run(config, limit=args.dry_run_limit), ensure_ascii=False, indent=2))
        return
    train(config)


if __name__ == "__main__":
    main()
