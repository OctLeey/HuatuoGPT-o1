# Clinical-RLVR Resume Bullets

## One-Line Project Name

Clinical-RLVR: Qwen3-based medical complex reasoning post-training system.

## Chinese Resume Version

- 基于 HuatuoGPT-o1 的 Medical Verifiable Problems 思路，构建 Qwen3 医疗复杂推理后训练系统，完成 MedQA/MedMCQA/PubMedQA 数据清洗、CoT SFT、GRPO/RLVR、自定义 reward 与自动化评测闭环。
- 设计答案正确性、格式合规、医学实体一致性和推理长度约束 reward，并通过 Base/SFT/GRPO 对比实验分析 reward hacking、幻觉实体、格式失效与过长推理问题。
- 使用 QLoRA 在 Qwen3-4B 上完成快速实验闭环，并迁移至 Qwen3-8B 进行正式实验；输出准确率、格式合规率、实体一致性、错误类型分布和失败案例分析。

## English Resume Version

- Built Clinical-RLVR, a HuatuoGPT-o1-based medical reasoning post-training pipeline for Qwen3, covering medical QA data construction, CoT SFT, GRPO/RLVR, custom reward design, generation, and offline evaluation.
- Designed answer correctness, format compliance, entity consistency, and length-control rewards, and analyzed reward hacking, hallucinated entities, invalid formats, and overlong reasoning through Base/SFT/GRPO comparisons.
- Ran Qwen3-4B QLoRA smoke experiments before scaling to Qwen3-8B, reporting accuracy, format rate, entity F1, reward trends, error taxonomy, and representative failure cases.

## Interview Talking Points

### Why HuatuoGPT-o1?

HuatuoGPT-o1 is useful because it frames medical reasoning as verifiable problem solving. It gives a concrete post-training recipe: construct verifiable medical problems, teach complex reasoning with SFT, then improve with verifier-based RL.

### What is my contribution?

The project adapts the idea to Qwen3 and turns it into a practical single-machine post-training workflow. The added work includes Qwen3 QLoRA SFT, GRPO/RLVR reward functions, entity consistency reward, offline evaluation, and error analysis.

### Why Qwen3?

Qwen3 is newer than the Qwen2.5 baseline used in HuatuoGPT-o1 variants, and it is a stronger fit for reasoning/post-training experiments. Qwen2.5 can still be used as a baseline if time allows.

### Why GRPO/RLVR instead of only SFT?

SFT teaches the model a format and style, but it does not directly optimize verifiable correctness. GRPO/RLVR uses answer correctness and auxiliary rewards to optimize generated reasoning behavior against verifiable signals.

### What can go wrong?

- The model may learn to satisfy the format without improving correctness.
- Entity reward may reward keyword overlap rather than true medical reasoning.
- Length penalty may over-compress reasoning if weighted too strongly.
- LLM-as-a-Judge evaluation can be biased and should be treated as auxiliary evidence.

### Honest Limits

This project evaluates medical exam-style reasoning, not clinical safety. Results should be reported as benchmark improvements and error analyses, not as claims of real-world diagnostic reliability.
