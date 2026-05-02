# Session Handoff 2026-05-02

This document is for another Codex session that needs to inspect, debug, or refactor the current code-scoring pipeline without re-deriving the active defaults from old experiment artifacts.

## Current Branch and Recent Direction

- Current working branch: `direct-final-free-review`
- Recent prompt/eval commits on this branch:
  - `041057d fix batch evaluator step context arg`
  - `b809fb1 use native reasoning for review steps`
  - `7db2649 structure step review prompts`
  - `c9b134f rewrite functional review prompts`
  - `b1d7a0f move generated stepwise final steps into instruction`
  - `76b9582 free direct final review from prefills`

Interpretation:
- The project is in the middle of a prompt-and-structure transition.
- Old explicit `<step>` tags are no longer the intended default for new reasoning trajectories.
- The current branch favors native reasoning notes plus a final `<review>` block.

## Active Project Goal

The current project goal is not generic code review generation. It is:

- Score candidate code primarily on functional correctness.
- Use AXIOM 0-5 as the internal scoring anchor.
- Treat textual comments as supporting evidence for the score, not the main output.

Current scope is deliberately narrowed:

- Default evaluation dimension is only `Correctness Verification`.
- The pipeline is optimized for code scoring, not multi-dimension review text generation.

## Default Model and Model Policy

Default maintained model:

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

Why this matters:

- Maintained wrappers now default to DeepSeek 7B rather than Qwen.
- Qwen wrappers are retained mainly for legacy comparison and debugging.

Relevant files:

- [README.md](/data1/xianzhiwei/mcts-code-review/README.md)
- [README.zh-CN.md](/data1/xianzhiwei/mcts-code-review/README.zh-CN.md)
- [deepseek7b_defaults.sh](/data1/xianzhiwei/mcts-code-review/data_collection/scripts/deepseek7b_defaults.sh)

Local-server default snapshot logic:

- `MODEL_KEY` defaults to `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `MODEL_PATH` prefers the local snapshot if present

## Default Data Assumptions

Primary datasets currently assumed by maintained workflows:

- `datasets/CodeCriticBench/data/CodeCriticBench.jsonl`
- `datasets/axiom-llm-judge/axiombench/*.jsonl`

Operational role split:

- CodeCriticBench: seed/source data for generation and training
- AXIOM: held-out evaluation target and scoring anchor

Current repo direction is AXIOM-aligned scoring, with CodeCriticBench mapped into that semantics.

## Default Scoring Semantics

Current scoring semantics are AXIOM-first:

- Internal grade: `0-5`
- Meaning: refinement effort with correctness-first split
- Grades `3-5`: functionally correct
- Grades `0-2`: functionally incorrect or fundamentally mismatched

Prompt-side wording is mirrored in:

- [prompt_template.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/prompt_template.py)
- [review_evaluator.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/review_evaluator.py)

Examples in the rubric are intentionally illustrative, not exhaustive.

## Current Default Inference and Eval Behavior

### Dimensions

Default eval dimension list:

- `["Correctness Verification"]`

Defined in:

- [review_evaluator.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/review_evaluator.py)

### Stepwise Context Mode

Current default:

- `step_context_mode = instruction_context`

Meaning:

- Previously generated reasoning notes are inserted into the instruction as `Previous analysis notes:`
- They are not continued as assistant-prefix text by default

This is now the maintained default in:

- [review_evaluator.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/review_evaluator.py)
- [batch_review_evaluator.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/batch_review_evaluator.py)
- [direct_bootstrap_review.py](/data1/xianzhiwei/mcts-code-review/data_collection/direct_bootstrap_review.py)

### Final-only vs Stepwise

There are two distinct eval modes and they must not be conflated:

- `final_only_json = 1`
  - Single final `<review>` turn
  - No intermediate reasoning steps during eval
- `final_only_json = 0`
  - Stepwise reasoning
  - Intermediate native reasoning notes
  - Final `<review>` at the last turn

Maintained wrapper behavior:

- `static` and `direct_review` evals are final-only
- `direct_stepwise_1step` / `direct_stepwise_2step` evals are stepwise

See:

- [run_direct_stepcount_vs_review_qwen3_4b.sh](/data1/xianzhiwei/mcts-code-review/data_collection/scripts/run_direct_stepcount_vs_review_qwen3_4b.sh)

### Tests in Prompt

Current default for evaluation:

- `show_tests_in_prompt = False`

Meaning:

- Dataset tests are hidden from the model unless explicitly enabled
- This is intended to keep evaluation closer to practical code scoring rather than oracle-assisted grading

### Stop Rules

Current stepwise/default stopping behavior:

- Intermediate reasoning stops on `</think>` or `</review>`
- Final generation stops on `</review>`

Review-MCTS config defaults also use:

- `stop: ["</think>", "</review>"]`

See:

- [mcts_code_review_deepseek_r1_distill_qwen_7b.yaml](/data1/xianzhiwei/mcts-code-review/data_collection/configs/mcts_code_review_deepseek_r1_distill_qwen_7b.yaml)

## Current Prompt Contract

### Intermediate reasoning

Current intended contract:

- Intermediate turns should generate one concise native reasoning note
- No XML `<step>` tags
- No JSON step objects
- No `<review>` in intermediate turns

Prompt definitions:

- [prompt_template.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/prompt_template.py)

### Final review

Current intended contract:

- Final answer must be exactly one `<review>...</review>` JSON block
- Required keys:
  - `axiom_grade`
  - `functional_correctness`
  - `repair_effort`
  - `evidence_type`
  - `summary`
  - `evidence`

Important branch nuance:

- Data-generation direct final prompts on this branch were relaxed so the model may reason before the last `<review>` block.
- Eval/final prompts are still stricter and expect a final parseable `<review>` block.

This mismatch is one of the active debugging risks.

## Current Default Generation and Training Workflows

### A. Direct step-count comparison workflow

Default maintained wrapper:

- [run_direct_stepcount_vs_review_deepseek7b.sh](/data1/xianzhiwei/mcts-code-review/data_collection/scripts/run_direct_stepcount_vs_review_deepseek7b.sh)

This wrapper currently means:

- Seed data from CodeCriticBench aligned to AXIOM
- Train four variants:
  - `static`
  - `direct_review`
  - `direct_stepwise_1step`
  - `direct_stepwise_2step`
- Evaluate on AXIOM held-out

Actual underlying implementation is inherited from:

- [run_direct_stepcount_vs_review_qwen3_4b.sh](/data1/xianzhiwei/mcts-code-review/data_collection/scripts/run_direct_stepcount_vs_review_qwen3_4b.sh)

### B. Review-MCTS generation

Primary entrypoint:

- [solver_review.py](/data1/xianzhiwei/mcts-code-review/data_collection/solver_review.py)

Primary config:

- [mcts_code_review_deepseek_r1_distill_qwen_7b.yaml](/data1/xianzhiwei/mcts-code-review/data_collection/configs/mcts_code_review_deepseek_r1_distill_qwen_7b.yaml)

Current config characteristics:

- `max_depth: 3`
- `n_generate_sample: 3`
- `iterations: 18`
- `review_explore_depth: 2`
- `review_target_leaf_count: 24`
- `need_value_func: False`
- `self_review_value_func: False`

Interpretation:

- Current review-MCTS path is still using reward computation from external sample/objective alignment, not a self-scoring value function during data generation.

### C. Training

Primary training entry:

- [train_multi.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/train_multi.py)

Current export/training design:

- Training items use chat-format `messages`
- Intermediate reasoning is placed inside `<think> ... </think>`
- Final answer is `<review> ... </review>`
- `assistant_parts` carry `q_value` labels for value supervision at segment boundaries

Primary preprocessing:

- [preprocess_review_mcts_data.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/preprocess_review_mcts_data.py)

## Current Important Sample/Eval Output Semantics

A frequent confusion point:

- Per-sample eval outputs store the real result under `dimensions[i]`, not top-level `final_review`

Relevant files:

- [batch_review_evaluator.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/batch_review_evaluator.py)
- [review_evaluator.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/review_evaluator.py)

Useful interpretation:

- `dimensions[0].trace`
  - per-step generation trace
- `dimensions[0].trace[*].candidates[*].continuation`
  - raw model continuation for that call
- `dimensions[0].trace[*].candidates[*].reasoning`
  - extracted native reasoning
- `dimensions[0].final_review`
  - final accumulated text after all accepted steps/retries
- `dimensions[0].final_review_parse`
  - parse result for final `<review>`
- `dimensions[0].final_retries`
  - extra final-review repair attempts after parse failure

## Known Current Problems

These are active issues, not historical curiosities:

1. The repo is still structurally bloated.
- There are many legacy wrappers, old prompt backups, and old experiment paths.
- New sessions can easily read the wrong workflow from old files.

2. Prompt behavior is in transition.
- Old `<step>`-tag logic and new native-reasoning logic coexist.
- Some preprocessors still keep backward-compatibility code for `<step>` samples.

3. Final review parseability is unstable after the recent prompt rewrite.
- Recent long runs showed many samples with `missing_review_tags`.
- The failure mode is often: long freeform reasoning, but no final `<review>`.

4. Data-generation and eval prompt contracts may no longer be fully aligned.
- Direct final generation on this branch allows freeform reasoning before the last `<review>`.
- Eval still relies on strict recoverable final parsing.

5. Recent long experiment to inspect:
- Run name: `native_reasoning_prompt_long_20260502`
- Training finished, but eval was interrupted once by a missing CLI arg and then resumed manually
- The run should not be treated as a clean benchmark result

6. The current branch has likely over-optimized toward freeing the model from prefixes without yet recovering reliable final formatting.

## Recommended Entry Files for a New Debug/Refactor Session

Read these first, in order:

1. [README.md](/data1/xianzhiwei/mcts-code-review/README.md)
2. [docs/codex_change_log.md](/data1/xianzhiwei/mcts-code-review/docs/codex_change_log.md)
3. [model_training/src/magicoder/prompt_template.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/prompt_template.py)
4. [model_training/src/magicoder/review_evaluator.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/review_evaluator.py)
5. [data_collection/direct_bootstrap_review.py](/data1/xianzhiwei/mcts-code-review/data_collection/direct_bootstrap_review.py)
6. [data_collection/mcts_math/review_utils.py](/data1/xianzhiwei/mcts-code-review/data_collection/mcts_math/review_utils.py)
7. [model_training/src/magicoder/preprocess_review_mcts_data.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/preprocess_review_mcts_data.py)

Then inspect actual broken samples rather than reading more wrappers.

## Immediate Refactor Guidance

If another session is doing cleanup, the safest priorities are:

1. Separate maintained workflows from legacy experiments more explicitly.
2. Add a compact debug output mode for eval artifacts.
3. Make one prompt contract authoritative across:
- direct bootstrap generation
- stepwise eval
- training export
4. Reduce output JSON to the fields actually needed for current debugging.
5. Do not treat old `<step>`-tag behavior as the mainline design.

## Background Run Policy

Long jobs are expected to run in `tmux`.

Current ignored output roots:

- `data_collection/review_mcts_runs/`
- `model_training/review_mcts_train_data/`
- `model_training/src/output/`

Notifications:

- Many maintained wrappers try to `curl` ntfy on finish/failure
- Notification success is not a source of truth
- Real source of truth is log files plus `comparison.json` or `summary.json`
