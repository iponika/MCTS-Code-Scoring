# CodeCritic More-Data Experiment Log

Date: 2026-05-14

This document is the running provenance log for the expanded CodeCriticBench experiment. Record every material decision here so the paper can trace data size, filtering, prompts, model checkpoints, and evaluation settings.

## Current Worktrees

There are currently two relevant worktrees on this server.

| Role | Path | Git state | Purpose |
|---|---|---|---|
| New/current worktree | `/data1/xianzhiwei/mcts-code-review` | branch `qwen3-4b-CCB`, commit `20d9bec82787923a2e22039d3ddb19c1aeec755b` | Current code after promptfix and value-sampling changes. |
| Old-prompt worktree | `/tmp/cc-oldprompt-ba27cf0` | detached `ba27cf0215a997cf48c003ae822fbb8b73e28911` | Reproduce the old-prompt experiment that produced the better CodeCritic retained-set results. |

The old-prompt worktree was created to avoid reverting the current branch while preserving the exact old code path for reruns.

## Baseline Old-Prompt Provenance

The detailed provenance for the old-prompt RQ1 experiment is in:

```text
docs/codecritic_qwen3_4b_oldprompt_experiment_provenance_20260514.md
```

Key old RQ1 split:

| Split | Count |
|---|---:|
| train optimistic seeds | 197 |
| test non-optimistic retained seeds | 859 |
| total eligible Easy non-QA seeds | 1056 |

## Prompt And Mechanism Differences

The main code delta is `ba27cf0..20d9bec`. The prompt/mechanism changes are concentrated in:

```text
shared/prompt_contract.py
data_collection/mcts_math/agents/utils.py
data_collection/mcts_math/agents/review_mcts.py
model_training/src/magicoder/preprocess_review_mcts_data.py
data_collection/scripts/run_codecritic_easy_qwen3_4b_all_optimistic_postprocess.sh
data_collection/scripts/run_codecritic_easy_qwen3_4b_mcts_worker.sh
data_collection/scripts/run_codecritic_easy_qwen3_4b_train_worker.sh
```

### Old-Prompt Worktree: `ba27cf0`

Prompt behavior:

- Intermediate MCTS step history opened a new explicit `<step>` slot after previous steps.
- The model was expected to generate a concise reasoning note before the final `<review>`.
- There was no explicit prompt instruction saying score-only output is invalid.
- Score-only intermediate outputs were not stripped or rejected at preprocessing time.
- Missing value outputs during MCTS expansion could terminalize candidate nodes more aggressively.

Training-data behavior:

- Optimistic seed selection used strict `over_count > under_count`.
- Policy paths used the +/-2 grade-delta rule.
- Value sampling kept up to 3 high-quality and 3 low-quality value paths per policy-producing record, aligned by available high-quality count.
- Value-only paths from records without policy paths were dropped when `--value_only_from_policy_records_only` was enabled.

### Old-Prompt Mainline Patch: `ba27cf0` + Step-Score Guard

After comparing old and current prompts, the expanded-data mainline keeps the old-prompt system and applies only a narrow prompt-only guard in:

```text
/tmp/cc-oldprompt-ba27cf0/shared/prompt_contract.py
```

Patch intent:

- If an earlier intermediate step includes a score or rating, tell the model to ignore that scoring act.
- Preserve and use the concrete reasoning/evidence from previous steps; only the previous step's scoring behavior is invalid.
- Tell the model not to copy, explain, reuse, continue, or treat the prior step score as a stopping signal.
- Apply the same step-score rule to final scoring prompts and training compatibility prompts, so generation and training prompts share the same interpretation of stale scored steps.
- Keep the old MCTS, parser, seed selection, value sampling, training, and evaluation mechanics unchanged.

Prompt cleanup:

- Removed the redundant final-prompt sentence `You don't have to re-analyze every detail by yourself.`
- Replaced it with the narrower prior-step-score instruction because the old sentence adds little useful constraint and can distract from evidence-based final scoring.
- Reworded one old training step-only sentence from `fixed context` / `earlier assistant messages` to `context, not facts`, which is less likely to preserve stale numeric outputs as evidence.

Recommended artifact label for future runs:

```text
oldprompt_step_score_guard_v1
```

Observed retained-set result from old-prompt full rerun on CodeCriticBench non-optimistic eval:

| Method | Alpha | QWK | ICC2 | RMSE | EMD | MAE |
|---|---:|---:|---:|---:|---:|---:|
| Base | 64.03 | 79.53 | 79.55 | 1.9372 | 1.1692 | 1.4119 |
| Static | 60.75 | 76.66 | 76.68 | 2.0035 | 1.3162 | 1.4749 |
| MCTS | 65.45 | 78.83 | 78.85 | 1.9396 | 1.0860 | 1.4026 |

Interpretation:

- MCTS beats Static clearly on agreement, calibration, and error metrics.
- MCTS improves over Base on Alpha, EMD, MAE, and mean signed error, but not on QWK/ICC2/RMSE/rank correlations.

### New/Current Worktree: `20d9bec`

Prompt behavior changes:

- Intermediate prompt history no longer opens a new raw `<step>` slot. It now ends with a numbered continuation prefix (`1. `) to avoid the model seeing a stop-suffix-like prompt ending.
- Prompt explicitly says: do not output only a number or score; if a prior intermediate turn accidentally output only a score, treat it as invalid and continue with concrete evidence.
- CodeCritic MCTS prompt now carries more explicit sample context into `review_context`: `scoring_target`, `problem`, `tests`, `dimension_rubrics`, and language fields.
- Step output parser rejects score-only outputs, strips score-only lines from otherwise useful reasoning, and drops empty parsed intermediate nodes.
- Final prompt in current code contains additional wording:
  - "You don't have to analyze code by yourself."
  - A hint about summarizing correctness status only when it helps introduce a new evidence check.

MCTS mechanism changes:

- When value outputs are missing for non-final candidate nodes, current MCTS assigns a neutral visit reward instead of immediately making the node terminal.
- Missing value on final candidates still receives negative reward and terminalization.
- Worker supports explicit resume control.

Training-data preprocessing changes:

- Score-only reasoning segments and score-only lines are removed from exported training examples.
- Terminal paths with parse errors/missing CodeCritic score are skipped for value sampling instead of being treated as ordinary incorrect value examples.
- If global sampled value positives exceed negatives, current code can borrow incorrect value paths from no-policy records and force them value-only to reduce imbalance.

Why this matters:

- These changes are defensible fixes for score-only degeneration and value imbalance.
- However, they also change the generated MCTS tree structure and the training-data distribution. The promptfix4 run produced fewer optimistic seeds and weaker retained-set results, so expanded-data experiments must record which worktree generated each artifact.

## Data-Size Check For Expansion

The RQ1 old Easy-only split is too small for a 2000+ sample target:

| Difficulty | Raw non-QA CodeGen rows | Eligible after Correctness Verification + conflict filtering |
|---|---:|---:|
| Easy | 1164 | 1056 |
| Medium field `Meidum` | 779 | 636 |
| Easy + Medium | 1943 | 1692 |
| Hard | 1257 | 939 |
| Easy + Medium + Hard | 3200 | 2631 |

Important dataset quirk:

```text
The CodeCriticBench difficulty value is spelled "Meidum", not "Medium".
```

Filtering rule used for the counts:

- source in `mbpp`, `codeforce`, `live-code-bench`, `debug`
- difficulty in the selected set
- has a usable `Correctness Verification` checklist score and checklist text
- drop label/score conflicts:
  - `Error` with target correctness score greater than 5
  - `Correct` with target correctness score less than 5

Conclusion:

- Easy + Medium gives only 1692 eligible samples, still below 2000.
- To exceed 2000 under the same filtering rule, the expanded experiment must include Hard, giving 2631 eligible samples.

## Open Decision For Expanded Run

The next experiment must choose the generation code path before launching MCTS:

| Option | Pros | Risks |
|---|---|---|
| Old-prompt `ba27cf0` | Best known retained-set results; stable comparison to old RQ1. | Keeps score-only/old MCTS quirks; harder to justify promptfix omission unless framed as empirical selection. |
| Current `20d9bec` | Cleaner prompt contract and preprocessing; fixes known degeneration modes. | Recent promptfix4 results were worse; may need more tuning before using for main paper result. |

Current recommendation:

Use old-prompt `ba27cf0` for the first expanded-data run if the immediate goal is to test whether more data improves the already-positive MCTS-vs-Static result. In parallel, preserve current-worktree results as an ablation of promptfix mechanisms rather than replacing the old-prompt mainline.

## Running Notes

- 2026-05-14: Created this log.
- 2026-05-14: Confirmed two worktrees: current `qwen3-4b-CCB` at `20d9bec`, old detached worktree at `ba27cf0`.
- 2026-05-14: Confirmed Easy+Medium is insufficient for 2000+ eligible samples; Easy+Medium+Hard gives 2631.
