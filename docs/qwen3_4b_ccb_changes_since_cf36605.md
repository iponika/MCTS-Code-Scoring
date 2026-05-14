# Qwen3-4B CodeCriticBench Changes Since cf36605

Date: 2026-05-14

Base commit:

```text
cf36605f291a62edf9d3b38212f3e8cd5aeb2520 add docs
```

Branch for these changes:

```text
qwen3-4b-CCB
```

These changes are kept on a separate branch because they alter shared CodeCritic data preparation, MCTS preprocessing, and evaluation behavior. They are not only server-local Qwen3-4B scripts, and therefore may change the replay protocol for the earlier Qwen3.5-9B experiment.

## Interface Impact

The changes affect the original Qwen3.5-9B CodeCritic experiment interface.

Affected shared paths:

- `data_collection/mcts_math/review_utils.py`
- `data_collection/prepare_codecritic_easy_correctness_splits.py`
- `model_training/src/magicoder/preprocess_review_mcts_data.py`
- `model_training/src/magicoder/review_evaluator.py`
- `docs/codecritic_qwen35_easy_mcts_experiment.md`

Reason:

- `target_correctness_score` extraction now uses the first raw `Correctness Verification` checklist score instead of allowing later duplicate dimensions to overwrite it.
- CodeCritic Easy split generation now filters label/score contradictions.
- CodeCritic value-only sampling now treats target +/- 2 as high-quality and caps low-quality count to the sampled high-quality count.
- CodeCritic final-review merging now uses the `correctness_score` prefill instead of the AXIOM `axiom_grade` prefill.
- The Qwen3.5 experiment document now reflects the corrected target-score and split semantics.

## Correctness Target Fix

Problem found:

Some CodeCriticBench rows contain duplicate `Correctness Verification` dimensions. The previous conversion used a dictionary keyed by dimension name, so later duplicate entries overwrote the first correctness score.

Example failure mode:

- Basic label: `Error`
- Overall score: low
- First `Correctness Verification` score: low and aligned with the bug
- Later duplicate `Correctness Verification` checklist score: high for a narrower checklist question
- Old `target_correctness_score`: high due to overwrite

Fix:

- `prepare_codecriticbench_sample()` now records the first `Correctness Verification` score as `target_correctness_score`.
- All raw correctness scores are preserved in `all_correctness_verification_scores` for audit.

Regression coverage:

- `tests/test_review_seed_preparation.py::test_prepare_codecriticbench_sample_uses_first_correctness_verification_score`

## Label/Score Conflict Filter

The CodeCritic Easy split builder now drops unsafe supervision rows:

```text
drop if correctness_label == Error and target_correctness_score > 5
drop if correctness_label == Correct and target_correctness_score < 5
```

Boundary score `5` is retained.

Corrected split counts for non-QA CodeGen Easy:

| Item | Count |
|---|---:|
| CodeGen Easy total | 1164 |
| Missing Correctness Verification advanced label | 57 |
| Label/score conflicts filtered | 51 |
| Eligible after filtering | 1056 |

Layer distribution after filtering:

| Layer | Count |
|---|---:|
| Low | 708 |
| Mid | 14 |
| High | 334 |

Regression coverage:

- `tests/test_review_seed_preparation.py::test_codecritic_correctness_filter_removes_label_score_conflicts`

## CodeCritic Static Data Fix

Static CodeCritic training items now emit the CodeCritic contract:

```json
{"correctness_score": <1-10 integer>, "dimension": "Correctness Verification", ...}
```

They no longer emit AXIOM `axiom_grade` for CodeCritic correctness samples.

Regression coverage:

- `tests/test_static_review_train_data.py`

## Evaluator Merge Fix

`review_evaluator.py` previously merged final continuations using the AXIOM prefill even when `--prompt_variant codecritic_correctness` was active. This could wrap CodeCritic generations as `axiom_grade` payloads.

Fix:

- CodeCritic eval now uses `CODECRITIC_FINAL_REVIEW_PREFILL`.
- Continuations such as `8, "dimension": "Correctness Verification", ...` are merged as `correctness_score`.

Regression coverage:

- `tests/test_review_evaluator_parsing.py::test_merge_final_review_continuation_uses_codecritic_prefill`

## Value Sampling Change

For CodeCritic MCTS preprocessing:

- Policy samples are still selected from terminal paths within target +/- 2.
- Value high-quality paths are now also defined as target +/- 2.
- Each policy-producing seed keeps at most 3 high-quality value paths.
- Low-quality value paths are capped to the number of sampled high-quality paths.

This prevents the previous behavior where exact-score value paths were scarce and low-quality paths dominated value supervision.

Regression coverage:

- `tests/test_review_training_dedupe.py::test_codecritic_value_sampling_treats_within_policy_delta_as_high_quality_and_aligns_low_count`

## Qwen3-4B Server Adaptation

Added local Qwen3-4B no-think support:

- `data_collection/configs/mcts_codecritic_qwen3_4b_no_think.yaml`
- `scripts/qwen3_4b_env.sh`
- `data_collection/scripts/run_codecritic_easy_qwen3_4b_mcts_worker.sh`
- `data_collection/scripts/run_codecritic_easy_qwen3_4b_train_worker.sh`
- `data_collection/scripts/run_codecritic_easy_qwen3_4b_eval_worker.sh`
- `data_collection/scripts/run_codecritic_easy_qwen3_4b_all_optimistic_postprocess.sh`

The environment script uses the local Qwen3-4B Hugging Face cache and keeps transient caches under `/tmp`.

## Optimistic Seed Experiment

Experiment document:

```text
docs/codecritic_qwen3_4b_optimistic_mcts_experiment_20260514.md
```

Run directory:

```text
data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514
```

All non-QA Easy seeds:

```text
seed_all_nonqa_easy.jsonl
```

Count:

```text
1056
```

MCTS generation:

```text
mcts_gpu0_000_528.jsonl
mcts_gpu1_528_528.jsonl
```

Optimistic seed rule:

```text
over_count  = count(predicted_correctness_score > target_correctness_score)
under_count = count(predicted_correctness_score < target_correctness_score)
keep seed iff over_count > under_count
```

Selection result:

| Item | Count |
|---|---:|
| Total MCTS records | 1056 |
| Optimistic records | 197 |
| Pessimistic records | 705 |
| Tied records | 154 |

Machine-readable selected seed list:

```text
data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514/optimistic_seed_indices.json
```

Training data:

| Dataset | Items |
|---|---:|
| MCTS optimistic policy/value | 629 |
| Static optimistic exact-label baseline | 197 |

MCTS optimistic training breakdown:

| Item | Count |
|---|---:|
| policy paths | 178 |
| exact policy paths | 81 |
| weak +/-2 policy paths | 97 |
| value-only paths | 451 |
| high-quality value paths | 404 |
| low-quality value paths | 47 |

Completed checkpoints:

```text
model_training/src/output/review-lora-qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514-mcts_optimistic-160step
model_training/src/output/review-lora-qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514-static_optimistic-160step
```

Training logs:

| Variant | Items after filtering | Truncated | Steps | Train loss |
|---|---:|---:|---:|---:|
| MCTS optimistic | 629 | 0 | 160 | 0.9916 |
| Static optimistic | 197 | 0 | 160 | 3.009 |

## Verification

Fresh verification run before commit:

```bash
PYTHONPATH=$PWD:$PWD/model_training/src:$PWD/data_collection python -m unittest \
  tests.test_review_training_dedupe \
  tests.test_review_seed_preparation \
  tests.test_static_review_train_data \
  tests.test_review_evaluator_parsing
```

Expected result:

```text
OK
```

Additional runtime checks performed:

- MCTS all-seed generation completed: `528 + 528 = 1056`.
- Optimistic postprocess completed and wrote `197` selected seeds.
- MCTS optimistic training completed with `160/160` steps.
- Static optimistic training completed with `160/160` steps.
