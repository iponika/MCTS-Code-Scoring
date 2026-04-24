# Bootstrap Data Comparison Design

**Date:** 2026-04-24

## Goal

Build a fair 4B comparison that isolates the effect of training-data generation strategy for code scoring:

- `Static-SFT`
- `Direct-Bootstrap-SFT`
- `MCTS-Bootstrap-SFT`

The primary claim is not "training helps", but "tree search generates better training data than non-tree direct sampling under the same scoring target and roughly matched generation budget."

## Scope

This design only covers the first complete comparison loop:

1. Prepare a balanced `CodeCriticBench -> AXIOM` seed set.
2. Generate bootstrap data by two methods:
   - direct independent rollouts
   - MCTS exploration
3. Convert both outputs into the same `train_multi` review-training format.
4. Build a static exact-label baseline from the same seed set.
5. Train the same 4B model with the same LoRA hyperparameters on each dataset.
6. Evaluate all trained checkpoints on the same AXIOM clean held-out set.

This design does not attempt to solve cross-dataset generalization fully. It only establishes a clean ablation for the data-generation stage.

## Fairness Constraints

The comparison must hold the following fixed:

- Same base model: `Qwen/Qwen3-4B-Instruct-2507`
- Same seed sample set
- Same scoring target: AXIOM 0-5
- Same reward/verifier logic for generated paths
- Same training script and main hyperparameters
- Same external evaluation set

The only intended variable is the data-generation strategy:

- no bootstrap (`Static-SFT`)
- direct linear bootstrap (`Direct-Bootstrap`)
- tree search bootstrap (`MCTS-Bootstrap`)

## Data Flow

### Seed Set

Use a balanced subset of `CodeCriticBench`, mapped to AXIOM grades and exported as normalized review samples.

Requirements:

- default to grades `1-5`
- equal count per grade
- preserve task, code, tests, and AXIOM target grade

### Static-SFT

Convert the normalized seed set directly into exact-label review training items:

- one instruction per sample
- one `<review>` response per sample
- exact value target from AXIOM grade

### Direct-Bootstrap

For each seed sample:

- generate multiple independent final reviews without tree search
- score every generated review with the same reward function used for MCTS leaves
- export a pseudo-`react` structure so existing preprocessing can be reused

Each direct candidate becomes a terminal-only linear path.

### MCTS-Bootstrap

Reuse the existing `solver_review.py` output format and preprocessing path.

## Reuse Strategy

To minimize new bugs:

- reuse `load_codecriticbench_dataset` for normalized sample loading
- reuse `compute_review_reward` for direct bootstrap leaf scoring
- reuse `preprocess_review_mcts_data.py` for both direct-bootstrap and MCTS-bootstrap exports
- reuse `train_multi.py` for all training groups
- reuse `run_axiom_clean_eval.sh` for external evaluation

The only new format bridge is a direct-bootstrap exporter that writes pseudo-tree records compatible with `preprocess_review_mcts_data.py`.

## Implementation Units

### 1. Seed-set builder

Create a dedicated script that writes balanced CodeCritic-only prepared review samples.

### 2. Direct-bootstrap generator

Create a local-vLLM generator that:

- loads the same seed set format as MCTS
- samples `K` independent final reviews
- computes reward for each candidate
- writes a pseudo-tree record with one terminal node per candidate

### 3. Static training-data converter

Create a converter from prepared review samples to `train_multi` JSONL so the static baseline uses the same prompt family and value scale.

### 4. 4B experiment runner

Create one orchestration script that:

- prepares the seed set
- runs direct bootstrap and MCTS bootstrap
- converts both outputs into train JSONL
- builds the static baseline train JSONL
- trains the three checkpoints with the same hyperparameters
- evaluates them on AXIOM clean held-out
- writes an aggregated summary

## Main Metrics

Primary comparison metrics:

- `grade_mae`
- `boundary_acc`
- `low_grade_false_positive_rate`
- `high_grade_false_negative_rate`
- `valid_rate`

Primary headline:

`MCTS-Bootstrap-SFT` vs `Direct-Bootstrap-SFT` on AXIOM clean held-out.

## Risks

- 4B generation and training still need conservative token budgets.
- Direct and MCTS generation budgets will not be perfectly token-equal in the first implementation; the runner should expose budget knobs explicitly so the next iteration can tighten alignment.
- Existing preprocessing assumes `react` records; direct bootstrap must mimic this format carefully.
