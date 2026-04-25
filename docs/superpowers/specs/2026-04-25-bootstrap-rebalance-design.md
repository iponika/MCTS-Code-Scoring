# Bootstrap Rebalance Design

## Goal

Reduce label noise in the fair `Direct-Bootstrap` vs `MCTS-Bootstrap` comparison without changing MCTS search or training code. The immediate target is to stop a small number of contradictory bootstrap leaves from dominating the value-only training set.

## Problem

The first formal 4B comparison showed that balanced sample counts were not enough. `MCTS` still underperformed because:

- strong AXIOM-boundary contradictions remained in the train set, especially `target 4/5 -> parsed <= 2` and `target 1/2 -> parsed >= 4`
- some noisy `dataset_index` values were repeated many more times than others after balancing
- value-only paths dominated training, so repeated contradictory leaves strongly affected model behavior

## Chosen Approach

Keep the existing pipeline and only tighten the rebalance stage:

1. Filter strong contradiction samples before the final rebalance export.
2. Rebalance with stratified round-robin sampling instead of pure random sampling.
3. Stratify bootstrap data by:
   - `dataset_index`
   - AXIOM delta bucket: `under_2plus`, `under_1`, `exact`, `over_1`, `over_2plus`, `unknown`
4. Preserve current `Static-SFT` behavior except optional dataset-aware spreading.

## Why This Approach

This is the smallest change that directly addresses the diagnosed failure mode. It does not touch:

- MCTS solver logic
- reward propagation
- training objective
- evaluator behavior

That keeps the next comparison interpretable: if results improve, the gain comes from better bootstrap data curation, not from changing multiple subsystems at once.

## Expected Outcome

- fewer extreme contradiction samples in the final bootstrap training JSONL
- less repeated amplification of a single noisy seed
- a cleaner comparison of whether MCTS can help once obviously harmful leaves are removed
