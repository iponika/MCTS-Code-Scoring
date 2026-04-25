# Bootstrap Rebalance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten bootstrap train-set rebalancing so fair `Direct-Bootstrap` vs `MCTS-Bootstrap` experiments no longer over-amplify strongly contradictory leaves.

**Architecture:** Extend the existing rebalance script with contradiction filtering and stratified round-robin sampling, then thread the new options through the 4B comparison runner. Verify the behavior first on existing raw JSONL outputs, then rerun a smaller formal comparison.

**Tech Stack:** Python, bash, JSONL training data, uv, existing 4B bootstrap comparison pipeline.

---

### Task 1: Extend the rebalance script

**Files:**
- Modify: `data_collection/rebalance_review_train_data.py`

- [ ] Add CLI flags for contradiction filtering and stratified sampling.
- [ ] Add AXIOM contradiction rules for `target 4/5 -> parsed <= 2` and `target 1/2 -> parsed >= 4`.
- [ ] Replace pure random sampling with grouped round-robin sampling over `dataset_index` and delta buckets.
- [ ] Emit summary fields for dropped contradictions and input/output delta distributions.

### Task 2: Thread the new behavior into the comparison runner

**Files:**
- Modify: `data_collection/scripts/run_bootstrap_comparison_qwen3_4b.sh`

- [ ] Add environment toggles for contradiction filtering and stratified sampling.
- [ ] Apply the new rebalance flags to bootstrap exports.
- [ ] Keep the static baseline compatible and avoid touching model training/eval stages.

### Task 3: Verify on existing formal raw outputs

**Files:**
- Reuse: `model_training/review_mcts_train_data/bootstrap_cmp_qwen3_4b_formal_20260424_*_raw.jsonl`

- [ ] Run the rebalance script on the existing formal raw direct export.
- [ ] Run the rebalance script on the existing formal raw MCTS export.
- [ ] Check that strong contradiction counts drop to zero in both outputs.
- [ ] Check that per-seed repetition is flatter than before.

### Task 4: Record the design and experiment-facing change

**Files:**
- Create: `docs/superpowers/specs/2026-04-25-bootstrap-rebalance-design.md`
- Create: `docs/superpowers/plans/2026-04-25-bootstrap-rebalance.md`
- Modify: `docs/codex_change_log.md`

- [ ] Summarize the diagnosed failure mode.
- [ ] Describe the new rebalance/filtering strategy.
- [ ] Record the new small formal rerun plan.

### Task 5: Run a smaller formal 4B comparison

**Files:**
- Reuse: `data_collection/scripts/run_bootstrap_comparison_qwen3_4b.sh`

- [ ] Start a new `tmux` run with the new rebalance logic enabled.
- [ ] Use a smaller seed budget and step budget than the first formal run for quick feedback.
- [ ] Inspect the resulting `summary.json` and compare against the previous formal run.
