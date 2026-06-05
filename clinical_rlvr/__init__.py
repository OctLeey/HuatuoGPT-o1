"""Utilities for the Clinical-RLVR HuatuoGPT-o1 extension."""

from .data import build_option_text, format_mcq_prompt, normalize_sample
from .datasets import build_dpo_record, build_rlvr_record, build_sft_record, load_raw_samples
from .evaluation import evaluate_records, summarize_error_types
from .generation import build_mock_output, build_mock_predictions, load_rlvr_records
from .judge import build_judge_prompt, mock_judge, parse_judge_label
from .rewards import (
    answer_reward,
    composite_train_reward,
    entity_reward,
    extract_final_answer,
    format_reward,
    length_penalty,
)

__all__ = [
    "answer_reward",
    "build_dpo_record",
    "build_judge_prompt",
    "build_option_text",
    "build_mock_output",
    "build_mock_predictions",
    "build_rlvr_record",
    "build_sft_record",
    "composite_train_reward",
    "entity_reward",
    "evaluate_records",
    "extract_final_answer",
    "format_mcq_prompt",
    "format_reward",
    "load_raw_samples",
    "load_rlvr_records",
    "length_penalty",
    "mock_judge",
    "normalize_sample",
    "parse_judge_label",
    "summarize_error_types",
]
