# Experiment Plan

## Current Direction

The project target is code scoring. Review text is auxiliary evidence for an AXIOM-style scalar grade.

Use objective labels for data generation:

- AXIOMBench exact 0-5 refinement-effort grades.
- CodeCriticBench mapped to AXIOM 0-5 through correctness boundary plus score buckets.
- MCTS path values from objective leaf rewards and backpropagation.
- No base-model text self-judging for node labels.

## Immediate Pipeline

1. Build a unified review-scoring JSONL from AXIOMBench and CodeCriticBench.
2. Run direct baseline on the same samples.
3. Run supervised review MCTS with only `Correctness Verification`.
4. Summarize `valid_rate`, `MAE`, `median_abs_error`, `boundary_accuracy`, and `pearson_proxy`.
5. Inspect extreme errors before using generated paths for training.

## Standard Medium Run

Prepare and run with tmux:

```bash
tmux new-session -d -s supervised_review_medium \
  'cd /data1/xianzhiwei/mcts-code-review && data_collection/scripts/run_supervised_review_medium.sh'
```

Monitor:

```bash
tail -f data_collection/review_mcts_runs/supervised_medium_*/run.log
nvidia-smi
```

The default medium dataset requests 13 CodeCritic samples per AXIOM grade and 50 AXIOM samples per grade, for about 378 samples. CodeCritic grade 2 is scarce, so 13 is the current balanced cap.

## Training Gate

Do not scale training until the medium run satisfies:

- MCTS output valid rate materially exceeds direct baseline.
- MCTS MAE is lower than direct baseline.
- MCTS Pearson proxy is higher than direct baseline.
- Boundary errors are explainable rather than systematic.

After this, use the generated MCTS paths for a small LoRA checkpoint before expanding data volume.

## Deprecated Paths

Historical `local_self_mcts_*` and `self_mcts_*` run directories were exploratory. They used base-model text self-judging and should not be used for new training data. Current code rejects `self_review_value_func: True` at runtime.
