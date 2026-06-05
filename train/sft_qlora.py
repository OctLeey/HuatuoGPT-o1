"""QLoRA SFT entrypoint for Clinical-RLVR.

This script trains on the `sft.json` format produced by
`scripts/build_clinical_rlvr_data.py`. It is intentionally separate from the
original HuatuoGPT-o1 SFT script, which targets full-parameter multi-GPU
training with Deepspeed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


REQUIRED_CONFIG_KEYS = {
    "model_name_or_path",
    "train_file",
    "output_dir",
    "max_seq_length",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config file."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as f:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required for YAML configs. Install pyyaml.") from exc
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required config keys."""

    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"missing required config keys: {', '.join(missing)}")
    train_file = Path(config["train_file"])
    if not train_file.exists():
        raise FileNotFoundError(f"train_file does not exist: {train_file}")


def load_sft_records(path: str | Path) -> list[dict[str, Any]]:
    """Load SFT records and validate the fields consumed by the trainer."""

    with Path(path).open(encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("SFT train_file must contain a JSON list")
    for idx, record in enumerate(records):
        if "prompt" not in record or "response" not in record:
            raise ValueError(f"SFT record {idx} must contain prompt and response")
    return records


@dataclass
class SftFeature:
    input_ids: list[int]
    labels: list[int]


class ClinicalSftDataset:
    """Minimal torch dataset for prompt/response SFT."""

    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_seq_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> SftFeature:
        record = self.records[idx]
        messages = record.get("messages")
        if messages:
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": record["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = record["prompt"]
            full_text = prompt_text + record["response"]

        input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]

        input_ids = input_ids[-self.max_seq_length :]
        labels = labels[-self.max_seq_length :]
        return SftFeature(input_ids=input_ids, labels=labels)


class SftDataCollator:
    """Pad input IDs and labels for causal LM SFT."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[SftFeature]) -> dict[str, Any]:
        import torch

        max_len = max(len(item.input_ids) for item in features)
        input_ids = []
        labels = []
        attention_mask = []
        pad_id = self.tokenizer.pad_token_id
        for item in features:
            pad_len = max_len - len(item.input_ids)
            input_ids.append(item.input_ids + [pad_id] * pad_len)
            labels.append(item.labels + [-100] * pad_len)
            attention_mask.append([1] * len(item.input_ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def dry_run(config: dict[str, Any]) -> dict[str, Any]:
    """Validate config and data without importing or loading model dependencies."""

    validate_config(config)
    records = load_sft_records(config["train_file"])
    return {
        "model_name_or_path": config["model_name_or_path"],
        "train_file": config["train_file"],
        "num_records": len(records),
        "output_dir": config["output_dir"],
        "max_seq_length": config["max_seq_length"],
    }


def train(config: dict[str, Any]) -> None:
    """Run QLoRA SFT."""

    validate_config(config)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(int(config.get("seed", 42)))
    records = load_sft_records(config["train_file"])

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if config.get("bnb_4bit_compute_dtype", "bfloat16") == "bfloat16" else torch.float16
    quantization_config = None
    if config.get("use_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    if config.get("use_4bit", True):
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=list(config.get("lora_target_modules", [])),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ClinicalSftDataset(records, tokenizer, int(config["max_seq_length"]))
    collator = SftDataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        run_name=config.get("run_name"),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
        warmup_ratio=float(config.get("warmup_ratio", 0.03)),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        num_train_epochs=float(config.get("num_train_epochs", 1)),
        max_steps=int(config.get("max_steps", -1)),
        bf16=bool(config.get("bf16", True)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        logging_steps=int(config.get("logging_steps", 10)),
        save_steps=int(config.get("save_steps", 100)),
        save_total_limit=int(config.get("save_total_limit", 2)),
        report_to=config.get("report_to", "none"),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Clinical-RLVR QLoRA SFT.")
    parser.add_argument("--config", required=True, help="YAML or JSON config file.")
    parser.add_argument("--dry_run", action="store_true", help="Validate config and data without loading a model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(dry_run(config), ensure_ascii=False, indent=2))
        return
    train(config)


if __name__ == "__main__":
    main()
