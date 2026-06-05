# Clinical-RLVR Experiment Results

## Setup

| Item | Value |
| --- | --- |
| Date |  |
| GPU |  |
| Base model |  |
| Dataset |  |
| Train size |  |
| Eval size |  |
| Max prompt length |  |
| Max completion length |  |
| Notes |  |

## Main Results

| Group | Model | Training | Accuracy | Format Rate | Judge Pass Rate | Entity F1 | Avg Think Tokens | Avg Reward |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | Qwen3-4B | none |  |  |  |  |  |  |
| SFT | Qwen3-4B | QLoRA SFT |  |  |  |  |  |  |
| GRPO-basic | Qwen3-4B | answer + format |  |  |  |  |  |  |
| GRPO-entity | Qwen3-4B | answer + format + entity |  |  |  |  |  |  |
| SFT | Qwen3-8B | QLoRA SFT |  |  |  |  |  |  |
| GRPO-entity | Qwen3-8B | answer + format + entity |  |  |  |  |  |  |

## Ablation

| Variant | Changed Component | Accuracy | Format Rate | Entity F1 | Avg Reward | Observation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| GRPO-basic | no entity reward |  |  |  |  |  |
| GRPO-entity | add entity reward |  |  |  |  |  |
| GRPO-short | stronger length penalty |  |  |  |  |  |

## Training Notes

- Loss/reward trend:
- Reward instability:
- Output format issues:
- Memory or runtime issues:
- Config changes:

## Conclusion

Summarize what improved, what did not improve, and which result is trustworthy enough to put on the resume.
