# Clinical-RLVR

Clinical-RLVR is a secondary development project based on HuatuoGPT-o1. It adapts the HuatuoGPT-o1 medical verifiable reasoning idea to Qwen3 and builds a practical post-training loop for medical multiple-choice reasoning.

The project focuses on a resume-ready post-training workflow:

- medical QA data construction
- CoT SFT data formatting
- Qwen3 QLoRA SFT
- GRPO/RLVR with custom rewards
- generation and offline evaluation
- error analysis and ablation reporting

This is a research and engineering project. It does not claim clinical safety or real-world medical reliability.

## What Changed

The original HuatuoGPT-o1 code provides paper-style SFT, PPO, verifier-based rewards, and evaluation. Clinical-RLVR keeps those files intact and adds an isolated extension:

```text
clinical_rlvr/
  data.py          # schema normalization and prompt formatting
  datasets.py      # SFT, DPO, and RLVR dataset builders
  rewards.py       # answer, format, entity, and length rewards
  generation.py    # mock and model prediction helpers
  evaluation.py    # offline metrics and error categories
scripts/
  build_clinical_rlvr_data.py
  generate_clinical_rlvr_outputs.py
  evaluate_clinical_rlvr_outputs.py
  judge_clinical_rlvr_outputs.py
train/
  sft_qlora.py
  grpo_rlvr.py
configs/
  clinical_rlvr_sft_qwen3_4b.yaml
  clinical_rlvr_sft_qwen3_8b.yaml
  clinical_rlvr_grpo_qwen3_4b.yaml
  clinical_rlvr_grpo_qwen3_8b.yaml
reports/
  experiment_results.md
  error_cases.md
  resume_bullets.md
```

## Pipeline

```text
raw medical QA
  -> normalized schema
  -> sft.json / dpo.json / rlvr.json
  -> Qwen3 QLoRA SFT
  -> generation
  -> offline evaluation
  -> GRPO/RLVR with custom rewards
  -> ablation and error analysis
```

## Quick Local Check

Build demo data:

```bash
python scripts/build_clinical_rlvr_data.py \
  --input data/demo_data.json \
  --output_dir output_data/clinical_rlvr_demo \
  --source demo \
  --limit 5
```

Validate the Qwen3-4B SFT config without loading a model:

```bash
python train/sft_qlora.py \
  --config configs/clinical_rlvr_sft_qwen3_4b.yaml \
  --dry_run
```

Run a mock generate/evaluate loop:

```bash
python scripts/generate_clinical_rlvr_outputs.py \
  --input output_data/clinical_rlvr_demo/rlvr.json \
  --output output_data/clinical_rlvr_demo/mock_predictions.json \
  --mock \
  --limit 5

python scripts/evaluate_clinical_rlvr_outputs.py \
  --input output_data/clinical_rlvr_demo/mock_predictions.json \
  --output output_data/clinical_rlvr_demo/mock_eval_report.json
```

Validate GRPO rewards without loading a model:

```bash
python train/grpo_rlvr.py \
  --config configs/clinical_rlvr_grpo_qwen3_4b.yaml \
  --dry_run \
  --dry_run_limit 3
```

Run tests:

```bash
python -m unittest tests.test_clinical_rlvr
```

Optional local mock judge:

```bash
python scripts/judge_clinical_rlvr_outputs.py \
  --input output_data/clinical_rlvr_demo/mock_predictions.json \
  --output output_data/clinical_rlvr_demo/mock_judged_predictions.json \
  --mock
```

## Training Plan

Start with Qwen3-4B:

```bash
python train/sft_qlora.py \
  --config configs/clinical_rlvr_sft_qwen3_4b.yaml
```

Then generate and evaluate SFT predictions:

```bash
python scripts/generate_clinical_rlvr_outputs.py \
  --input output_data/clinical_rlvr_demo/rlvr.json \
  --output output_data/clinical_rlvr_demo/qwen3_4b_sft_predictions.json \
  --model_name_or_path Qwen/Qwen3-4B \
  --adapter_path ckpts/clinical_rlvr_sft_qwen3_4b \
  --max_new_tokens 1024 \
  --temperature 0.2

python scripts/evaluate_clinical_rlvr_outputs.py \
  --input output_data/clinical_rlvr_demo/qwen3_4b_sft_predictions.json \
  --output output_data/clinical_rlvr_demo/qwen3_4b_sft_eval_report.json
```

After the SFT smoke run is stable, run GRPO/RLVR:

```bash
pip install -r requirements-clinical-rlvr-grpo.txt

python train/grpo_rlvr.py \
  --config configs/clinical_rlvr_grpo_qwen3_4b.yaml
```

Scale to Qwen3-8B only after the 4B pipeline is stable.

## Rewards

Training reward:

```text
R = 1.0 * answer_reward
  + 0.2 * format_reward
  + 0.3 * entity_reward
  - 0.1 * length_penalty
```

Current reward components:

- `answer_reward`: final option matches the reference answer key.
- `format_reward`: output contains valid `<think>` and `<answer>` blocks.
- `entity_reward`: lexical entity consistency between model reasoning/final answer and the reference answer/rationale.
- `length_penalty`: discourages overly long or repetitive reasoning.

The LLM-as-a-Judge verifier is planned for offline semantic evaluation and later ablation once the SFT/GRPO loop is stable.
The first implementation is available in `scripts/judge_clinical_rlvr_outputs.py`; it supports mock judging for local tests and OpenAI-compatible chat completion APIs with caching for real evaluation.

## Metrics

Offline evaluation reports:

- accuracy
- format rate
- judge pass rate, when judged predictions include `judge_score`
- average entity F1
- average reasoning tokens
- average reward
- error type counts
- per-sample details

Error categories include:

- correct
- invalid output format
- final answer wrong
- reasoning plausible but final option wrong
- answer correct but format invalid
- answer correct but medical reasoning weak
- overly long or repetitive reasoning

## Expected Experiments

Minimum comparison:

| Group | Model | Training | Notes |
| --- | --- | --- | --- |
| Base | Qwen3-4B | none | first generation baseline |
| SFT | Qwen3-4B | QLoRA SFT | validates data and format learning |
| GRPO-basic | Qwen3-4B | answer + format reward | first RLVR baseline |
| GRPO-entity | Qwen3-4B | answer + format + entity reward | tests entity reward |
| SFT | Qwen3-8B | QLoRA SFT | formal larger-model run |
| GRPO-entity | Qwen3-8B | custom rewards | formal RLVR run |

Use `reports/experiment_results.md` and `reports/error_cases.md` to record final results.

## Relationship to HuatuoGPT-o1

HuatuoGPT-o1 uses medical verifiable problems, complex reasoning SFT, and PPO with a medical verifier. Clinical-RLVR borrows that direction but changes the engineering target:

- Qwen3 is used as the main model family.
- QLoRA is used for practical single-machine training.
- GRPO/RLVR custom rewards are added for answer correctness, format compliance, entity consistency, and reasoning length.
- Offline evaluation and error analysis are added for resume-ready reporting.

Original HuatuoGPT-o1 files remain available as the paper baseline.
