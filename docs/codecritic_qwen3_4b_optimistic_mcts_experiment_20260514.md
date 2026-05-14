# Qwen3-4B CodeCritic Easy Optimistic MCTS Seed Experiment

Date: 2026-05-14

This document records the optimistic-seed data selection and training-data generation for the Qwen3-4B no-think CodeCriticBench Easy correctness experiment.

## Source Data

- Run directory: `data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514`
- Base seed file: `seed_all_nonqa_easy.jsonl`
- Base seed count: `1056` non-QA CodeGen Easy rows after filtering missing Correctness Verification labels and label/score conflicts.
- MCTS shards: `mcts_gpu0_000_528.jsonl` and `mcts_gpu1_528_528.jsonl`, `528 + 528 = 1056` records.
- Model: `Qwen/Qwen3-4B`, no-think mode, CodeCritic correctness prompt, visible tests enabled.

## Seed Filtering Contract

- `target_correctness_score` uses the first raw `Correctness Verification` checklist score.
- Rows are excluded when `correctness_label == Error` and target score is greater than 5, or when `correctness_label == Correct` and target score is less than 5.
- Optimistic seed selection is applied after full MCTS generation, not before generation.

## Optimistic Selection Rule

For each MCTS tree, terminal CodeCritic review leaves are compared with the seed target score:

```text
over_count  = count(predicted_correctness_score > target_correctness_score)
under_count = count(predicted_correctness_score < target_correctness_score)
exact_count = count(predicted_correctness_score == target_correctness_score)
keep seed iff over_count > under_count
```

## Selection Summary

| Item | Count |
|---|---:|
| records | 1056 |
| optimistic_records | 197 |
| pessimistic_records | 705 |
| tied_records | 154 |
| seed_rows | 197 |

### Optimistic Seed Distribution

**Source**

| Value | Count |
|---|---:|
| codeforce | 53 |
| debug | 14 |
| live-code-bench | 25 |
| mbpp | 105 |

**Score Layer**

| Value | Count |
|---|---:|
| high | 65 |
| low | 126 |
| mid | 6 |

**Correctness Label**

| Value | Count |
|---|---:|
| Correct | 69 |
| Error | 128 |

**Target Correctness Score**

| Value | Count |
|---|---:|
| 1.0 | 10 |
| 2.0 | 80 |
| 3.0 | 24 |
| 4.0 | 13 |
| 5.0 | 3 |
| 6.0 | 10 |
| 7.0 | 19 |
| 8.0 | 14 |
| 9.0 | 24 |

## Training Data

- Training JSONL: `mcts_optimistic_policy_pm2_value_pm2_aligned_train.jsonl`
- Training stats: `mcts_optimistic_policy_pm2_value_pm2_aligned_train_stats.json`
- Policy selection: terminal paths within target +/- 2.
- Value sampling: high-quality value paths are also target +/- 2; at most 3 high-quality and low-quality paths per policy-producing record; if high-quality paths are fewer than 3, low-quality count is capped to the sampled high-quality count.

| Item | Count |
|---|---:|
| final_policy_paths | 178 |
| final_policy_exact_grade_paths | 81 |
| final_policy_weak_grade_delta_paths | 97 |
| final_value_only_paths | 451 |
| output_items | 629 |
| new_records_without_policy | 19 |
| new_value_sampled_correct_paths | 404 |
| new_value_sampled_incorrect_paths | 47 |

## Selected Optimistic Seed Indices

The same table is also available as machine-readable JSON at `optimistic_seed_indices.json`.

| dataset_index | original_dataset_index | source | target | label | layer | over | under | exact | leaves |
|---:|---:|---|---:|---|---|---:|---:|---:|---:|
| 1 | 1 | mbpp | 6.0 | Correct | high | 3 | 0 | 0 | 3 |
| 4 | 4 | mbpp | 6.0 | Correct | high | 4 | 0 | 0 | 4 |
| 11 | 11 | mbpp | 7.0 | Correct | high | 4 | 0 | 2 | 6 |
| 12 | 12 | mbpp | 6.0 | Correct | mid | 1 | 0 | 0 | 1 |
| 14 | 14 | mbpp | 6.0 | Correct | high | 3 | 0 | 0 | 3 |
| 16 | 16 | mbpp | 7.0 | Correct | high | 2 | 0 | 0 | 2 |
| 19 | 20 | mbpp | 7.0 | Correct | high | 5 | 0 | 0 | 5 |
| 21 | 22 | mbpp | 8.0 | Correct | high | 2 | 0 | 0 | 2 |
| 23 | 24 | mbpp | 9.0 | Correct | high | 3 | 0 | 0 | 3 |
| 26 | 28 | mbpp | 9.0 | Correct | high | 6 | 0 | 0 | 6 |
| 27 | 29 | mbpp | 5.0 | Correct | mid | 5 | 0 | 0 | 5 |
| 31 | 33 | mbpp | 8.0 | Correct | high | 5 | 0 | 0 | 5 |
| 32 | 34 | mbpp | 9.0 | Correct | high | 4 | 0 | 1 | 5 |
| 33 | 35 | mbpp | 8.0 | Correct | high | 1 | 0 | 3 | 4 |
| 35 | 37 | mbpp | 9.0 | Correct | high | 2 | 0 | 2 | 4 |
| 37 | 39 | mbpp | 6.0 | Correct | high | 4 | 0 | 0 | 4 |
| 38 | 40 | mbpp | 6.0 | Correct | high | 3 | 0 | 0 | 3 |
| 39 | 42 | mbpp | 7.0 | Correct | high | 3 | 0 | 0 | 3 |
| 41 | 44 | mbpp | 9.0 | Correct | high | 3 | 0 | 0 | 3 |
| 42 | 45 | mbpp | 9.0 | Correct | high | 1 | 0 | 0 | 1 |
| 46 | 49 | mbpp | 5.0 | Correct | high | 3 | 0 | 0 | 3 |
| 49 | 52 | mbpp | 8.0 | Correct | high | 3 | 0 | 0 | 3 |
| 51 | 54 | mbpp | 7.0 | Correct | high | 2 | 1 | 1 | 4 |
| 53 | 56 | mbpp | 8.0 | Correct | high | 6 | 1 | 0 | 7 |
| 55 | 58 | mbpp | 9.0 | Correct | high | 3 | 0 | 0 | 3 |
| 60 | 63 | mbpp | 6.0 | Correct | mid | 11 | 0 | 0 | 11 |
| 61 | 64 | mbpp | 7.0 | Correct | high | 2 | 0 | 0 | 2 |
| 62 | 65 | mbpp | 7.0 | Correct | high | 1 | 0 | 0 | 1 |
| 64 | 67 | mbpp | 6.0 | Correct | high | 2 | 0 | 0 | 2 |
| 66 | 69 | mbpp | 7.0 | Correct | mid | 3 | 0 | 0 | 3 |
| 67 | 71 | mbpp | 7.0 | Correct | high | 2 | 1 | 2 | 5 |
| 68 | 72 | mbpp | 6.0 | Correct | high | 3 | 0 | 0 | 3 |
| 70 | 74 | mbpp | 8.0 | Correct | high | 4 | 0 | 0 | 4 |
| 71 | 75 | mbpp | 8.0 | Correct | high | 2 | 0 | 0 | 2 |
| 72 | 76 | mbpp | 9.0 | Correct | high | 3 | 0 | 0 | 3 |
| 73 | 79 | mbpp | 3.0 | Error | low | 2 | 0 | 1 | 3 |
| 74 | 80 | mbpp | 2.0 | Error | low | 3 | 0 | 0 | 3 |
| 75 | 81 | mbpp | 2.0 | Error | low | 2 | 0 | 1 | 3 |
| 77 | 83 | mbpp | 2.0 | Error | low | 4 | 0 | 0 | 4 |
| 79 | 85 | mbpp | 4.0 | Error | low | 3 | 0 | 0 | 3 |
| 80 | 86 | mbpp | 3.0 | Error | low | 3 | 2 | 4 | 9 |
| 82 | 88 | mbpp | 2.0 | Error | low | 3 | 0 | 0 | 3 |
| 86 | 92 | mbpp | 4.0 | Error | low | 2 | 1 | 0 | 3 |
| 87 | 93 | mbpp | 2.0 | Error | low | 3 | 1 | 1 | 5 |
| 88 | 94 | mbpp | 3.0 | Error | low | 2 | 0 | 1 | 3 |
| 95 | 101 | mbpp | 2.0 | Error | low | 2 | 1 | 1 | 4 |
| 98 | 104 | mbpp | 2.0 | Error | low | 2 | 0 | 4 | 6 |
| 99 | 105 | mbpp | 2.0 | Error | low | 3 | 1 | 1 | 5 |
| 102 | 108 | mbpp | 2.0 | Error | low | 1 | 0 | 3 | 4 |
| 103 | 110 | mbpp | 2.0 | Error | low | 3 | 0 | 3 | 6 |
| 108 | 115 | mbpp | 3.0 | Error | low | 4 | 2 | 1 | 7 |
| 111 | 118 | mbpp | 2.0 | Error | low | 6 | 0 | 0 | 6 |
| 113 | 120 | mbpp | 3.0 | Error | low | 1 | 0 | 3 | 4 |
| 116 | 123 | mbpp | 2.0 | Error | low | 3 | 1 | 0 | 4 |
| 120 | 127 | mbpp | 3.0 | Error | low | 2 | 1 | 1 | 4 |
| 125 | 132 | mbpp | 2.0 | Error | low | 2 | 0 | 1 | 3 |
| 128 | 135 | mbpp | 4.0 | Error | low | 4 | 0 | 0 | 4 |
| 130 | 137 | mbpp | 4.0 | Error | low | 2 | 0 | 0 | 2 |
| 131 | 139 | mbpp | 2.0 | Error | low | 1 | 0 | 3 | 4 |
| 132 | 140 | mbpp | 1.0 | Error | low | 2 | 0 | 0 | 2 |
| 138 | 147 | mbpp | 1.0 | Error | low | 1 | 0 | 5 | 6 |
| 145 | 155 | mbpp | 2.0 | Error | low | 2 | 0 | 0 | 2 |
| 146 | 156 | mbpp | 1.0 | Error | low | 1 | 0 | 4 | 5 |
| 147 | 157 | mbpp | 2.0 | Error | low | 1 | 0 | 3 | 4 |
| 149 | 159 | mbpp | 3.0 | Error | low | 7 | 0 | 0 | 7 |
| 159 | 169 | mbpp | 2.0 | Error | low | 1 | 0 | 0 | 1 |
| 167 | 179 | mbpp | 3.0 | Error | low | 3 | 0 | 8 | 11 |
| 169 | 181 | mbpp | 2.0 | Error | low | 6 | 0 | 1 | 7 |
| 170 | 182 | mbpp | 2.0 | Error | low | 4 | 0 | 0 | 4 |
| 176 | 189 | mbpp | 4.0 | Error | low | 5 | 0 | 0 | 5 |
| 178 | 191 | mbpp | 2.0 | Error | low | 2 | 0 | 1 | 3 |
| 181 | 194 | mbpp | 2.0 | Error | low | 1 | 0 | 1 | 2 |
| 182 | 195 | mbpp | 1.0 | Error | low | 2 | 0 | 1 | 3 |
| 197 | 213 | mbpp | 4.0 | Error | low | 5 | 0 | 0 | 5 |
| 201 | 217 | mbpp | 2.0 | Error | low | 5 | 0 | 1 | 6 |
| 207 | 225 | mbpp | 3.0 | Error | low | 2 | 0 | 3 | 5 |
| 208 | 226 | mbpp | 2.0 | Error | low | 6 | 0 | 0 | 6 |
| 209 | 227 | mbpp | 1.0 | Error | low | 4 | 0 | 5 | 9 |
| 212 | 230 | mbpp | 3.0 | Error | low | 1 | 0 | 3 | 4 |
| 214 | 232 | mbpp | 2.0 | Error | low | 3 | 0 | 0 | 3 |
| 216 | 235 | mbpp | 1.0 | Error | low | 1 | 0 | 2 | 3 |
| 217 | 237 | mbpp | 4.0 | Error | low | 3 | 1 | 0 | 4 |
| 220 | 241 | mbpp | 3.0 | Error | low | 5 | 0 | 0 | 5 |
| 222 | 243 | mbpp | 2.0 | Error | low | 2 | 0 | 0 | 2 |
| 226 | 247 | mbpp | 3.0 | Error | low | 3 | 0 | 3 | 6 |
| 231 | 255 | mbpp | 4.0 | Error | low | 5 | 0 | 0 | 5 |
| 232 | 256 | mbpp | 2.0 | Error | low | 1 | 0 | 0 | 1 |
| 236 | 260 | mbpp | 4.0 | Error | low | 3 | 2 | 0 | 5 |
| 237 | 261 | mbpp | 2.0 | Error | low | 6 | 0 | 3 | 9 |
| 238 | 262 | mbpp | 3.0 | Error | low | 3 | 0 | 6 | 9 |
| 241 | 265 | mbpp | 3.0 | Error | low | 8 | 0 | 0 | 8 |
| 243 | 269 | mbpp | 2.0 | Error | low | 3 | 0 | 1 | 4 |
| 244 | 270 | mbpp | 4.0 | Error | low | 6 | 0 | 0 | 6 |
| 246 | 272 | mbpp | 4.0 | Error | low | 8 | 2 | 0 | 10 |
| 266 | 296 | mbpp | 4.0 | Error | low | 9 | 0 | 0 | 9 |
| 274 | 304 | mbpp | 5.0 | Error | low | 2 | 0 | 0 | 2 |
| 275 | 305 | mbpp | 2.0 | Error | low | 1 | 0 | 2 | 3 |
| 276 | 306 | mbpp | 3.0 | Error | low | 2 | 0 | 4 | 6 |
| 277 | 307 | mbpp | 2.0 | Error | low | 4 | 0 | 4 | 8 |
| 278 | 308 | mbpp | 3.0 | Error | low | 1 | 0 | 1 | 2 |
| 279 | 309 | mbpp | 3.0 | Error | low | 8 | 0 | 0 | 8 |
| 283 | 313 | mbpp | 3.0 | Error | low | 2 | 0 | 8 | 10 |
| 287 | 317 | mbpp | 2.0 | Error | low | 3 | 0 | 1 | 4 |
| 289 | 319 | mbpp | 2.0 | Error | low | 3 | 0 | 0 | 3 |
| 298 | 328 | mbpp | 3.0 | Error | low | 3 | 0 | 0 | 3 |
| 313 | 343 | codeforce | 9.0 | Correct | high | 2 | 0 | 0 | 2 |
| 318 | 348 | codeforce | 7.0 | Correct | high | 2 | 1 | 0 | 3 |
| 329 | 359 | codeforce | 9.0 | Correct | high | 3 | 0 | 0 | 3 |
| 332 | 362 | codeforce | 7.0 | Correct | high | 2 | 0 | 0 | 2 |
| 339 | 369 | codeforce | 9.0 | Correct | high | 4 | 1 | 0 | 5 |
| 347 | 377 | codeforce | 9.0 | Correct | high | 3 | 0 | 0 | 3 |
| 353 | 383 | codeforce | 8.0 | Correct | high | 3 | 0 | 1 | 4 |
| 358 | 388 | codeforce | 9.0 | Correct | high | 3 | 0 | 0 | 3 |
| 361 | 391 | codeforce | 8.0 | Correct | high | 2 | 0 | 3 | 5 |
| 363 | 393 | codeforce | 9.0 | Correct | high | 3 | 1 | 3 | 7 |
| 364 | 394 | codeforce | 9.0 | Correct | high | 2 | 1 | 2 | 5 |
| 365 | 395 | codeforce | 9.0 | Correct | high | 4 | 0 | 0 | 4 |
| 367 | 397 | codeforce | 7.0 | Correct | high | 3 | 1 | 0 | 4 |
| 368 | 398 | codeforce | 9.0 | Correct | high | 1 | 0 | 3 | 4 |
| 369 | 399 | codeforce | 8.0 | Correct | high | 1 | 0 | 3 | 4 |
| 371 | 401 | codeforce | 9.0 | Correct | high | 1 | 0 | 0 | 1 |
| 372 | 402 | codeforce | 8.0 | Correct | high | 4 | 0 | 0 | 4 |
| 373 | 403 | codeforce | 8.0 | Correct | high | 2 | 0 | 1 | 3 |
| 376 | 406 | codeforce | 9.0 | Correct | high | 4 | 0 | 0 | 4 |
| 405 | 439 | codeforce | 2.0 | Error | low | 1 | 0 | 2 | 3 |
| 419 | 453 | codeforce | 2.0 | Error | low | 5 | 0 | 0 | 5 |
| 422 | 456 | codeforce | 2.0 | Error | low | 4 | 0 | 0 | 4 |
| 434 | 470 | codeforce | 1.0 | Error | low | 2 | 0 | 0 | 2 |
| 437 | 473 | codeforce | 2.0 | Error | low | 3 | 0 | 1 | 4 |
| 438 | 474 | codeforce | 2.0 | Error | low | 4 | 0 | 0 | 4 |
| 442 | 478 | codeforce | 2.0 | Error | low | 2 | 0 | 2 | 4 |
| 450 | 487 | codeforce | 2.0 | Error | low | 3 | 0 | 0 | 3 |
| 459 | 497 | codeforce | 2.0 | Error | low | 1 | 0 | 3 | 4 |
| 461 | 499 | codeforce | 3.0 | Error | low | 1 | 0 | 2 | 3 |
| 470 | 510 | codeforce | 2.0 | Error | low | 3 | 0 | 3 | 6 |
| 476 | 516 | codeforce | 3.0 | Error | low | 3 | 2 | 1 | 6 |
| 479 | 519 | codeforce | 2.0 | Error | low | 4 | 0 | 0 | 4 |
| 484 | 525 | codeforce | 2.0 | Error | low | 4 | 0 | 0 | 4 |
| 489 | 530 | codeforce | 2.0 | Error | low | 2 | 0 | 2 | 4 |
| 490 | 531 | codeforce | 2.0 | Error | low | 2 | 0 | 1 | 3 |
| 494 | 535 | codeforce | 2.0 | Error | low | 2 | 0 | 5 | 7 |
| 502 | 543 | codeforce | 1.0 | Error | low | 2 | 0 | 0 | 2 |
| 513 | 554 | codeforce | 2.0 | Error | low | 4 | 0 | 0 | 4 |
| 518 | 559 | codeforce | 2.0 | Error | low | 4 | 0 | 0 | 4 |
| 522 | 563 | codeforce | 2.0 | Error | low | 3 | 0 | 0 | 3 |
| 527 | 568 | codeforce | 1.0 | Error | low | 2 | 0 | 1 | 3 |
| 528 | 569 | codeforce | 2.0 | Error | low | 5 | 0 | 0 | 5 |
| 532 | 573 | codeforce | 2.0 | Error | low | 2 | 0 | 2 | 4 |
| 540 | 582 | codeforce | 2.0 | Error | low | 3 | 0 | 5 | 8 |
| 547 | 589 | codeforce | 2.0 | Error | low | 2 | 0 | 0 | 2 |
| 552 | 595 | codeforce | 2.0 | Error | low | 4 | 0 | 0 | 4 |
| 556 | 599 | codeforce | 2.0 | Error | low | 2 | 0 | 2 | 4 |
| 560 | 603 | codeforce | 2.0 | Error | low | 2 | 0 | 0 | 2 |
| 562 | 606 | codeforce | 3.0 | Error | low | 2 | 0 | 7 | 9 |
| 564 | 608 | codeforce | 2.0 | Error | low | 2 | 0 | 3 | 5 |
| 576 | 620 | codeforce | 3.0 | Error | low | 9 | 0 | 0 | 9 |
| 580 | 624 | codeforce | 2.0 | Error | low | 2 | 0 | 0 | 2 |
| 586 | 630 | codeforce | 2.0 | Error | low | 5 | 0 | 0 | 5 |
| 609 | 655 | live-code-bench | 9.0 | Correct | high | 6 | 0 | 0 | 6 |
| 611 | 657 | live-code-bench | 9.0 | Correct | high | 6 | 0 | 0 | 6 |
| 624 | 670 | live-code-bench | 8.0 | Correct | high | 1 | 0 | 0 | 1 |
| 636 | 682 | live-code-bench | 8.0 | Correct | high | 2 | 0 | 2 | 4 |
| 660 | 706 | live-code-bench | 7.0 | Correct | high | 2 | 0 | 2 | 4 |
| 662 | 708 | live-code-bench | 9.0 | Correct | high | 4 | 1 | 0 | 5 |
| 676 | 722 | live-code-bench | 7.0 | Correct | high | 5 | 0 | 0 | 5 |
| 698 | 744 | live-code-bench | 6.0 | Correct | high | 1 | 0 | 0 | 1 |
| 718 | 764 | live-code-bench | 9.0 | Correct | high | 4 | 0 | 0 | 4 |
| 725 | 771 | live-code-bench | 9.0 | Correct | high | 3 | 1 | 0 | 4 |
| 730 | 776 | live-code-bench | 7.0 | Correct | high | 4 | 0 | 1 | 5 |
| 742 | 788 | live-code-bench | 7.0 | Correct | high | 5 | 0 | 0 | 5 |
| 746 | 792 | live-code-bench | 7.0 | Correct | high | 3 | 0 | 0 | 3 |
| 748 | 794 | live-code-bench | 7.0 | Correct | high | 7 | 0 | 0 | 7 |
| 749 | 795 | live-code-bench | 7.0 | Correct | high | 6 | 0 | 0 | 6 |
| 751 | 798 | live-code-bench | 2.0 | Error | low | 4 | 0 | 0 | 4 |
| 757 | 805 | live-code-bench | 2.0 | Error | low | 4 | 0 | 1 | 5 |
| 764 | 813 | live-code-bench | 2.0 | Error | low | 1 | 0 | 5 | 6 |
| 782 | 833 | live-code-bench | 2.0 | Error | low | 2 | 0 | 1 | 3 |
| 787 | 838 | live-code-bench | 2.0 | Error | low | 5 | 0 | 1 | 6 |
| 796 | 849 | live-code-bench | 2.0 | Error | low | 3 | 0 | 3 | 6 |
| 800 | 853 | live-code-bench | 2.0 | Error | low | 1 | 0 | 1 | 2 |
| 801 | 854 | live-code-bench | 2.0 | Error | low | 1 | 0 | 0 | 1 |
| 814 | 868 | live-code-bench | 2.0 | Error | low | 5 | 0 | 0 | 5 |
| 823 | 878 | live-code-bench | 2.0 | Error | low | 1 | 0 | 4 | 5 |
| 875 | 934 | debug | 3.0 | Error | low | 2 | 0 | 3 | 5 |
| 892 | 961 | debug | 2.0 | Error | low | 7 | 0 | 0 | 7 |
| 916 | 994 | debug | 4.0 | Error | low | 3 | 1 | 0 | 4 |
| 932 | 1014 | debug | 2.0 | Error | low | 3 | 0 | 1 | 4 |
| 936 | 1019 | debug | 2.0 | Error | low | 3 | 1 | 1 | 5 |
| 940 | 1026 | debug | 2.0 | Error | low | 3 | 0 | 1 | 4 |
| 984 | 1075 | debug | 2.0 | Error | low | 2 | 0 | 1 | 3 |
| 1006 | 1101 | debug | 1.0 | Error | low | 3 | 0 | 0 | 3 |
| 1019 | 1119 | debug | 2.0 | Error | low | 1 | 0 | 4 | 5 |
| 1032 | 1133 | debug | 2.0 | Error | low | 5 | 0 | 0 | 5 |
| 1049 | 1155 | debug | 2.0 | Error | low | 2 | 0 | 0 | 2 |
| 1050 | 1156 | debug | 2.0 | Error | low | 3 | 0 | 2 | 5 |
| 1054 | 1162 | debug | 2.0 | Error | mid | 4 | 0 | 0 | 4 |
| 1055 | 1163 | debug | 2.0 | Error | mid | 2 | 0 | 3 | 5 |
