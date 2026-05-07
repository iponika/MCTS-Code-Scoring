# Session Handoff 2026-05-07

This document is for another Codex session on another server. It summarizes the current maintained experiment line, the latest results that actually matter, and the files/scripts that should be treated as the main entrypoints.

## Current Active Task

The current active task is:

- Compare `Static`, `Direct Review`, `1-step`, `2-step`, and `3-step` training on AXIOM-only data.
- Use `Qwen/Qwen3-4B` in `no_think` mode.
- Judge code primarily by functional correctness under AXIOM 0-5 semantics.
- Use this same-distribution AXIOM run to decide whether explicit stepwise reasoning helps code scoring.

This is the current mainline because:

- CodeCriticBench does not strictly match AXIOM semantics.
- AXIOM labels are cleaner for the current scoring objective.
- The latest prompt/eval repairs have finally made stepwise outputs parseable enough to compare on the same distribution.

## Current Branch and Relevant Commits

- Current working branch: `direct-final-free-review`
- Latest task-specific commits:
  - `a1324ce add axiom direct stepcount overnight pipeline`
  - `bd00a82 fix direct stepcount summary import path`

Interpretation:

- `a1324ce` adds the AXIOM-only seed builder and the maintained overnight wrapper for this comparison line.
- `bd00a82` fixes the last-stage summary import path bug. Before that fix, generation, training, and evaluation completed, but the wrapper failed when writing `summary.json`.

## Current Default Model for This Task

For the current maintained comparison task, the default model is:

- `Qwen/Qwen3-4B`

and the maintained config is:

- [mcts_code_review_qwen3_4b_no_think.yaml](/data1/xianzhiwei/mcts-code-review/data_collection/configs/mcts_code_review_qwen3_4b_no_think.yaml)

Important clarification:

- DeepSeek 7B wrappers still exist in the repo.
- They are not the current default for the active AXIOM step-count comparison task.
- If another session is continuing the current mainline, start from Qwen3-4B `no_think`, not DeepSeek.

## Current Dataset Policy

### Training / generation source

- AXIOM only
- Source directory:
  - [axiombench](/data1/xianzhiwei/mcts-code-review/datasets/axiom-llm-judge/axiombench)

### Held-out evaluation

- AXIOM held-out
- Training seed rows are excluded from held-out construction

### Why this replaced CodeCriticBench for the current mainline

- AXIOM labels are already in the target scoring semantics.
- CodeCriticBench can still be useful later as an aligned auxiliary dataset, but not as the cleanest current benchmark source.
- Direct inspection of sampled rows showed AXIOM is materially cleaner than CodeCriticBench for strict AXIOM-style supervision.

## Current Scoring Semantics

The project is currently narrowed to code scoring, not broad review generation.

Primary objective:

- Functional correctness scoring

Internal score anchor:

- AXIOM `0-5`

Practical interpretation:

- `3-5`: functionally correct
- `0-2`: functionally incorrect or fundamentally mismatched

Textual review is supporting evidence for the score, not the primary project output.

## Current Prompt / Eval Contract

### Stepwise intermediate reasoning

Current intent:

- Stepwise training/eval uses explicit numbered reasoning.
- Intermediate reasoning is still carried through the assistant-side response path.
- The current step experiments intentionally bias the model toward visible numbered continuation.

### Final review turn

Current maintained final-turn behavior:

- Prior reasoning is moved into the instruction, not continued as assistant-prefix history.
- If prior reasoning exists, the final prompt tells the model it does not have to analyze the code by itself again.
- If there are zero prior steps, that extra hint is omitted.

Reason:

- This change repaired the earlier failure mode where stepwise final turns kept drifting into more prose instead of emitting a parseable final review.

### Output contract

Maintained final output:

- one `<review>...</review>` JSON block

Core keys:

- `axiom_grade`
- `functional_correctness`
- `repair_effort`
- `evidence_type`
- `summary`
- `evidence`

## Current Maintained Workflow

### 1. Build AXIOM seed set

Seed builder:

- [prepare_axiom_seedset.py](/data1/xianzhiwei/mcts-code-review/data_collection/prepare_axiom_seedset.py)

What it does:

- Reads AXIOM rows directly from `datasets/axiom-llm-judge/axiombench`
- Builds balanced prepared samples by grade
- Preserves provenance metadata

### 2. Run the maintained direct step-count comparison

Main wrapper:

- [run_direct_stepcount_axiom_qwen3_4b.sh](/data1/xianzhiwei/mcts-code-review/data_collection/scripts/run_direct_stepcount_axiom_qwen3_4b.sh)

Underlying workflow:

- [run_direct_stepcount_vs_review_qwen3_4b.sh](/data1/xianzhiwei/mcts-code-review/data_collection/scripts/run_direct_stepcount_vs_review_qwen3_4b.sh)

Current maintained setup:

- `Static`
- `Direct Review`
- `1-step`
- `2-step`
- `3-step`

### 3. Training

Primary training entry:

- [train_multi.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/train_multi.py)

Current exported data behavior:

- Policy/value training items are written under:
  - [review_mcts_train_data](/data1/xianzhiwei/mcts-code-review/model_training/review_mcts_train_data)
- Checkpoints are written under:
  - [output](/data1/xianzhiwei/mcts-code-review/model_training/src/output)

### 4. Evaluation

Held-out AXIOM eval is still driven by:

- [run_axiom_clean_eval.sh](/data1/xianzhiwei/mcts-code-review/data_collection/scripts/run_axiom_clean_eval.sh)
- [review_evaluator.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/review_evaluator.py)
- [batch_review_evaluator.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/batch_review_evaluator.py)

## Current Best-Available Result

The current source of truth is:

- [summary.json](/data1/xianzhiwei/mcts-code-review/data_collection/review_mcts_runs/direct_stepcount_axiom_qwen3_4b_nothink_overnight_20260506/summary.json)

Run directory:

- [direct_stepcount_axiom_qwen3_4b_nothink_overnight_20260506](/data1/xianzhiwei/mcts-code-review/data_collection/review_mcts_runs/direct_stepcount_axiom_qwen3_4b_nothink_overnight_20260506)

Key metrics on AXIOM held-out:

| Method | valid_rate | MAE | median_err | boundary_acc |
|---|---:|---:|---:|---:|
| Static | 1.0000 | 1.8200 | 2.0 | 0.5000 |
| Direct Review | 0.9800 | 1.5714 | 1.0 | 0.5510 |
| 1-step | 0.9600 | 1.5417 | 1.0 | 0.4792 |
| 2-step | 1.0000 | 1.8400 | 2.0 | 0.4200 |
| 3-step | 0.9800 | 1.7143 | 1.0 | 0.4694 |

Current interpretation:

- `1-step` is the best current variant if the target metric is scalar grade MAE.
- `Direct Review` is the best current variant if the target is safer correctness-boundary behavior.
- `2-step` is clearly weak and should not be treated as a promising default.
- `3-step` recovers slightly from `2-step` but still does not beat `1-step` or `Direct Review`.

If another session needs a concise conclusion:

- The current useful comparison is `Direct Review` vs `1-step`.
- Do not spend more time on `2-step` or `3-step` unless the prompt contract changes substantially.

## Current Training Data Reality

For the latest AXIOM overnight run:

- `Static`: 150 total, 150 policy
- `Direct Review`: 152 total, 27 policy
- `1-step`: 600 total, 31 policy
- `2-step`: 600 total, 28 policy
- `3-step`: 600 total, 16 policy

Interpretation:

- Stepwise variants still have many more value-only rows than policy rows.
- The current stepwise success is therefore limited and fragile.
- If another session tries to improve results, focus on increasing useful policy-quality final reviews, not merely adding more steps.

## Important Files to Read First on Another Server

Read these first, in order:

1. [README.md](/data1/xianzhiwei/mcts-code-review/README.md)
2. [README.zh-CN.md](/data1/xianzhiwei/mcts-code-review/README.zh-CN.md)
3. [docs/codex_change_log.md](/data1/xianzhiwei/mcts-code-review/docs/codex_change_log.md)
4. [prepare_axiom_seedset.py](/data1/xianzhiwei/mcts-code-review/data_collection/prepare_axiom_seedset.py)
5. [run_direct_stepcount_axiom_qwen3_4b.sh](/data1/xianzhiwei/mcts-code-review/data_collection/scripts/run_direct_stepcount_axiom_qwen3_4b.sh)
6. [run_direct_stepcount_vs_review_qwen3_4b.sh](/data1/xianzhiwei/mcts-code-review/data_collection/scripts/run_direct_stepcount_vs_review_qwen3_4b.sh)
7. [prompt_contract.py](/data1/xianzhiwei/mcts-code-review/shared/prompt_contract.py)
8. [review_evaluator.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/review_evaluator.py)
9. [train_multi.py](/data1/xianzhiwei/mcts-code-review/model_training/src/magicoder/train_multi.py)

## Known Current Risks

1. The repo still contains older DeepSeek wrappers and older transition-era prompt experiments.
- Do not infer the active default from file count.
- Infer it from this handoff plus the latest change log.

2. Stepwise prompting is still an experiment, not a settled architecture.
- The current useful signal is only for `1-step`.
- More steps do not currently help.

3. Summary/log truth hierarchy still matters.
- `summary.json`, held-out comparison files, and raw eval artifacts are the source of truth.
- Notifications are secondary.

4. If a wrapper fails only at the end, inspect whether generation/training/eval already completed before re-running the whole job.
- This just happened in the AXIOM overnight run and was fixed by `bd00a82`.

## Background Run Policy

Maintained long runs should use `tmux`.

Useful habits:

- Verify `tmux ls`
- Check recent log lines with `tail`
- Check GPU state with `nvidia-smi`
- Check artifact growth with `wc -l` or file counts

On a new server, prefer resuming from this AXIOM-only direct-vs-stepcount line before reopening older MCTS or DeepSeek branches.
