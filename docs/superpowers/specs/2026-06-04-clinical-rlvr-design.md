# Clinical-RLVR Design

## Project Goal

Clinical-RLVR is a secondary development project based on HuatuoGPT-o1. It adapts the HuatuoGPT-o1 medical verifiable reasoning pipeline to Qwen3 and builds a practical post-training workflow for medical QA reasoning.

The project is intended to produce a GitHub-ready and resume-ready post-training project in 3-4 weeks. It should demonstrate data construction, CoT SFT, preference/RL training, reward design, evaluation, ablation, and error analysis.

## Assumptions

- The project will be developed on top of the HuatuoGPT-o1 codebase.
- AutoDL rental GPUs with at least 24GB VRAM are available.
- Full-parameter training is out of scope. LoRA or QLoRA will be used.
- Qwen3 is the main model family. Qwen2.5 is used only as an optional baseline.
- The first release prioritizes a complete and verifiable post-training loop over very large-scale training.

## Scope

### In Scope

- Build unified medical verifiable QA datasets from MedQA, MedMCQA, and PubMedQA.
- Convert medical multiple-choice QA into structured prompts with verifiable final answers.
- Build CoT SFT data with `<think>` and `<answer>` formatting.
- Build DPO chosen/rejected pairs from correct answers, wrong options, and model-generated hard negatives.
- Implement GRPO/RLVR-style training with reward functions.
- Add two focused improvements over a plain HuatuoGPT-o1 reproduction:
  - LLM-as-a-Judge verifier for semantic medical answer evaluation.
  - Medical entity consistency reward.
- Compare Base, SFT, DPO, GRPO-basic, and GRPO-entity variants.
- Produce experiment reports, ablation results, and error-case analysis.

### Out of Scope

- Full PRM or step-level process reward modeling in the first release.
- Search-based CoT generation at HuatuoGPT-o1 paper scale.
- Training 32B, 70B, or MoE models.
- Building a deployed medical assistant product.
- Claiming clinical safety or real-world medical reliability.

## Model Plan

- Fast iteration model: `Qwen3-4B`.
- Formal experiment model: `Qwen3-8B`.
- Optional baseline: `Qwen2.5-7B-Instruct` or `Qwen2.5-7B`.

Training should use LoRA or QLoRA. The Qwen3-4B stage validates data, rewards, and evaluation before spending GPU budget on Qwen3-8B.

## Architecture

The project is organized into five modules:

1. Data construction
2. SFT training
3. DPO and GRPO/RLVR training
4. Evaluation
5. Reporting and resume artifacts

High-level pipeline:

```text
Raw medical QA data
  -> preprocessing and unified schema
  -> medical verifiable problems
  -> SFT data / DPO pairs / RLVR prompts
  -> Qwen3-4B fast experiments
  -> Qwen3-8B formal experiments
  -> evaluation, ablation, and error analysis
```

## Data Schema

Each processed sample should use a unified structure:

```json
{
  "id": "medqa_000001",
  "source": "MedQA",
  "question": "...",
  "options": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  },
  "answer_key": "C",
  "answer": "...",
  "rationale": "...",
  "verifiable_answer": "...",
  "task_type": "multiple_choice_medical_qa"
}
```

The schema should support SFT, DPO, GRPO/RLVR, and evaluation without separate ad hoc formats.

## Training Data Construction

### SFT Data

SFT samples should teach the model to produce stable medical reasoning format:

```text
<think>
...
</think>
<answer>
C
</answer>
```

If the original dataset includes rationales, use them after cleaning. If not, generate a limited number of rationales with a stronger model and filter them before training.

### DPO Data

DPO pairs should be constructed as:

- `chosen`: correct answer with compliant reasoning format.
- `rejected`: wrong option answer, malformed answer, or model-sampled incorrect answer.

Hard negatives should include answers with plausible medical language but wrong final options or inconsistent entities.

### RLVR Prompts

RLVR prompts should contain the question and options only. The model generates reasoning and final answer, then rewards are computed from the output and reference fields.

## Reward Design

Training reward:

```text
R_train = R_answer + 0.2 * R_format + 0.3 * R_entity - 0.1 * R_length
```

Evaluation reward and metrics also include judge-based semantic correctness:

```text
R_eval = accuracy + judge_pass_rate + entity_consistency + format_rate
```

### Answer Reward

For multiple-choice QA:

- `1.0` if the final answer option matches `answer_key`.
- `0.0` otherwise.

This is the main verifiable signal for RLVR.

### Format Reward

Reward compliant output format:

- `1.0` for complete `<think>` and `<answer>` blocks.
- `0.5` for extractable final answer without full reasoning format.
- `0.0` for invalid or unextractable output.

### LLM-as-a-Judge Verifier

The judge verifier compares the question, reference answer, and model answer, then returns `correct`, `incorrect`, or `uncertain`.

This verifier is used primarily for offline evaluation and ablation. It may be used in training only if latency and cost are acceptable.

### Entity Consistency Reward

The entity reward compares medical entities in the reference answer/rationale and the model answer. Entity categories include:

- disease
- symptom
- drug
- examination
- treatment
- anatomy

The first implementation may use LLM extraction with caching. The score is based on entity F1, with optional penalties for clearly conflicting entities.

### Length Penalty

The length penalty discourages overly long or repetitive reasoning:

- No penalty below the target reasoning length.
- Light penalty for moderately long reasoning.
- Stronger penalty for very long or repetitive reasoning.

The penalty must be weak enough that the model does not collapse to short unsupported answers.

## Experiment Matrix

| Group | Model | Training | Purpose |
| --- | --- | --- | --- |
| Base | Qwen3-8B | none | raw baseline |
| SFT | Qwen3-8B | CoT QLoRA SFT | learn medical reasoning format |
| DPO | Qwen3-8B or Qwen3-4B | chosen/rejected preference data | preference optimization baseline |
| GRPO-basic | Qwen3-8B | answer + format reward | RLVR baseline |
| GRPO-entity | Qwen3-8B | answer + format + entity reward | test entity reward improvement |
| Eval-judge | all groups | offline judge evaluation | semantic correctness analysis |

If GPU budget is tight, DPO can be run only on Qwen3-4B, while Qwen3-8B focuses on Base, SFT, GRPO-basic, and GRPO-entity.

## Metrics

- Accuracy
- Format compliance rate
- Judge pass rate
- Entity F1
- Average reasoning tokens
- Error type distribution

Error categories:

- final answer wrong
- reasoning plausible but final option wrong
- answer correct but medical reasoning wrong
- invalid output format
- hallucinated medical entity
- overly long or repetitive reasoning

## Expected Repository Structure

```text
clinical-rlvr/
  README.md
  configs/
    sft_qwen3_4b.yaml
    sft_qwen3_8b.yaml
    grpo_qwen3_8b.yaml
  data/
    README.md
  scripts/
    build_dataset.py
    build_dpo_pairs.py
    run_eval.py
  src/
    rewards/
      answer_reward.py
      format_reward.py
      entity_reward.py
      judge_verifier.py
    data/
      preprocess.py
    eval/
      metrics.py
      error_analysis.py
  train/
    sft.py
    dpo.py
    grpo.py
  reports/
    experiment_results.md
    error_cases.md
```

When implemented inside HuatuoGPT-o1, this structure should be adapted to the existing repository layout rather than forced wholesale.

## Schedule

### Week 1

- Read HuatuoGPT-o1 paper and code.
- Prepare unified dataset schema.
- Build SFT, DPO, and RLVR data builders.
- Add small local tests for parsing and reward calculation.

### Week 2

- Run Qwen3-4B SFT smoke experiment.
- Run small DPO or GRPO experiment.
- Validate evaluation pipeline and reward behavior.

### Week 3

- Run Qwen3-8B QLoRA SFT.
- Run GRPO-basic and GRPO-entity formal experiments.
- Collect training curves and evaluation outputs.

### Week 4

- Complete ablation and error analysis.
- Write README, model card, experiment report, and resume bullets.
- Prepare interview explanation for data, reward, training, and failure cases.

## Resume Summary Draft

Clinical-RLVR: Qwen3-based medical complex reasoning post-training system.

Built a HuatuoGPT-o1-based medical RLVR pipeline with MedQA/MedMCQA/PubMedQA data construction, CoT SFT, DPO/GRPO training, entity-aware reward design, and LLM-as-a-Judge verifier evaluation. Compared Base/SFT/DPO/GRPO variants on accuracy, format compliance, semantic correctness, entity consistency, and error types, and analyzed reward hacking, hallucinated entities, and overlong reasoning.

## Verification Criteria

The project is considered successful when:

- The repository contains runnable data construction, reward, training, and evaluation scripts.
- Qwen3-4B completes at least one end-to-end smoke experiment.
- Qwen3-8B completes at least SFT and one GRPO/RLVR experiment.
- Evaluation outputs include Base, SFT, and GRPO comparison.
- Reports include ablation for entity reward or judge evaluation.
- README and reports describe limits honestly without claiming clinical reliability.
