# Clinical-RLVR AutoDL Notes

This note records the first runnable training path for the Clinical-RLVR extension. The goal is to run Qwen3 QLoRA SFT after data construction is verified locally.

## 1. Environment

Install the project requirements on the AutoDL machine:

```bash
pip install -r requirements.txt
```

The Clinical-RLVR QLoRA script additionally relies on:

- `datasets`
- `peft`
- `bitsandbytes`
- `pyyaml`

These are listed in `requirements.txt`.

## 2. Build SFT Data

Start with demo data or a small medical QA subset:

```bash
python scripts/build_clinical_rlvr_data.py \
  --input data/demo_data.json \
  --output_dir output_data/clinical_rlvr_demo \
  --source demo \
  --limit 100
```

This writes:

```text
output_data/clinical_rlvr_demo/normalized.json
output_data/clinical_rlvr_demo/sft.json
output_data/clinical_rlvr_demo/dpo.json
output_data/clinical_rlvr_demo/rlvr.json
output_data/clinical_rlvr_demo/summary.json
```

## 3. Dry Run

Validate config and data before loading the model:

```bash
python train/sft_qlora.py \
  --config configs/clinical_rlvr_sft_qwen3_4b.yaml \
  --dry_run
```

Expected output includes the model name, train file, record count, output directory, and max sequence length.

## 4. Qwen3-4B Smoke SFT

Run the first GPU smoke experiment:

```bash
python train/sft_qlora.py \
  --config configs/clinical_rlvr_sft_qwen3_4b.yaml
```

Use this stage to check:

- loss decreases rather than becoming NaN
- output checkpoint is saved under `ckpts/clinical_rlvr_sft_qwen3_4b`
- generated outputs keep the `<think>` and `<answer>` format
- offline evaluation can read predictions and compute metrics

## 5. Qwen3-8B Formal SFT

After the 4B smoke run succeeds:

```bash
python train/sft_qlora.py \
  --config configs/clinical_rlvr_sft_qwen3_8b.yaml
```

The 8B config uses larger gradient accumulation and a lower learning rate. Adjust batch size and sequence length if GPU memory is tight.

## 6. Generate Predictions

Local mock generation for pipeline validation:

```bash
python scripts/generate_clinical_rlvr_outputs.py \
  --input output_data/clinical_rlvr_demo/rlvr.json \
  --output output_data/clinical_rlvr_demo/mock_predictions.json \
  --mock \
  --limit 20
```

Real generation from a base model:

```bash
python scripts/generate_clinical_rlvr_outputs.py \
  --input output_data/clinical_rlvr_demo/rlvr.json \
  --output output_data/clinical_rlvr_demo/qwen3_4b_base_predictions.json \
  --model_name_or_path Qwen/Qwen3-4B \
  --max_new_tokens 1024 \
  --temperature 0.2 \
  --limit 200
```

Real generation from a QLoRA adapter:

```bash
python scripts/generate_clinical_rlvr_outputs.py \
  --input output_data/clinical_rlvr_demo/rlvr.json \
  --output output_data/clinical_rlvr_demo/qwen3_4b_sft_predictions.json \
  --model_name_or_path Qwen/Qwen3-4B \
  --adapter_path ckpts/clinical_rlvr_sft_qwen3_4b \
  --max_new_tokens 1024 \
  --temperature 0.2 \
  --limit 200
```

## 7. Offline Evaluation

Evaluate generated predictions:

```bash
python scripts/evaluate_clinical_rlvr_outputs.py \
  --input output_data/clinical_rlvr_demo/mock_predictions.json \
  --output output_data/clinical_rlvr_demo/mock_eval_report.json
```

The report includes:

```text
accuracy
format_rate
judge_pass_rate
avg_entity_f1
avg_reasoning_tokens
avg_reward
error_types
details
```

For the first real experiment, compare at least:

- Qwen3-4B base predictions
- Qwen3-4B SFT predictions
- Qwen3-8B base predictions
- Qwen3-8B SFT predictions

Optional LLM-as-a-Judge labeling:

```bash
python scripts/judge_clinical_rlvr_outputs.py \
  --input output_data/clinical_rlvr_demo/qwen3_4b_sft_predictions.json \
  --output output_data/clinical_rlvr_demo/qwen3_4b_sft_judged.json \
  --cache output_data/clinical_rlvr_demo/judge_cache.json \
  --model_name gpt-4o-mini \
  --api_key "$OPENAI_API_KEY"

python scripts/evaluate_clinical_rlvr_outputs.py \
  --input output_data/clinical_rlvr_demo/qwen3_4b_sft_judged.json \
  --output output_data/clinical_rlvr_demo/qwen3_4b_sft_judged_eval_report.json
```

For local testing without an API:

```bash
python scripts/judge_clinical_rlvr_outputs.py \
  --input output_data/clinical_rlvr_demo/mock_predictions.json \
  --output output_data/clinical_rlvr_demo/mock_judged_predictions.json \
  --mock
```

## 8. GRPO / RLVR Training

The original HuatuoGPT-o1 repository provides a PPO script based on a verifier reward model. Clinical-RLVR adds a separate GRPO/RLVR path using custom answer, format, entity, and length rewards.

Install the extra GRPO dependencies on AutoDL:

```bash
pip install -r requirements-clinical-rlvr-grpo.txt
```

Validate the GRPO config and reward functions without loading the model:

```bash
python train/grpo_rlvr.py \
  --config configs/clinical_rlvr_grpo_qwen3_4b.yaml \
  --dry_run
```

Run a Qwen3-4B GRPO smoke experiment:

```bash
python train/grpo_rlvr.py \
  --config configs/clinical_rlvr_grpo_qwen3_4b.yaml
```

After the 4B experiment is stable, run the 8B configuration:

```bash
python train/grpo_rlvr.py \
  --config configs/clinical_rlvr_grpo_qwen3_8b.yaml
```

For the first GRPO run, keep the dataset small and inspect:

- reward totals are not all identical
- answer reward is the dominant signal
- format reward does not overpower answer correctness
- entity reward improves explanation consistency without causing verbose repetition
- KL/beta is high enough to avoid the model drifting into reward hacking

## 9. Notes

- Keep generated data, checkpoints, logs, and wandb output out of git.
- The original HuatuoGPT-o1 `SFT_stage1.py` remains available as the paper-style full training baseline.
- The new `train/sft_qlora.py` is the practical single-machine Qwen3 LoRA/QLoRA path for this project.
- The new `train/grpo_rlvr.py` is the custom reward GRPO/RLVR path and uses a separate requirements file because the paper PPO baseline pins an older TRL version.
- Do not claim clinical reliability; this is a medical reasoning post-training experiment.
