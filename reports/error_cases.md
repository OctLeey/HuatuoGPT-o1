# Clinical-RLVR Error Cases

Use this file to record representative cases after each evaluation run. Prefer concrete examples over vague summaries.

## Error Taxonomy

| Error Type | Meaning | Count |
| --- | --- | ---: |
| final_answer_wrong | Extracted final option is wrong |  |
| reasoning_plausible_but_final_option_wrong | Format and reasoning look plausible, but final option is wrong |  |
| answer_correct_but_format_invalid | Answer is correct but output format breaks parsing |  |
| answer_correct_but_medical_reasoning_weak | Final option is correct but explanation misses key medical evidence |  |
| invalid_output_format | No extractable final answer |  |
| overly_long_or_repetitive_reasoning | Reasoning is too long or loops |  |

## Case Template

### Case 1

| Field | Value |
| --- | --- |
| Source |  |
| Question ID |  |
| Reference Answer |  |
| Predicted Answer |  |
| Error Type |  |
| Model Variant |  |

Question:

```text

```

Model Output:

```text

```

Analysis:

- What went wrong:
- Which reward/metric detected it:
- Possible fix:

## Summary

- Most frequent error:
- Most important reward hacking pattern:
- Best qualitative improvement after SFT:
- Best qualitative improvement after GRPO:
- Remaining risk:
