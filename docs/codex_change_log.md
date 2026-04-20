# Codex Change Log

This file records Codex-made project changes so work can be resumed safely across forks and sessions. Each future code or workflow change should update this log and be followed by a git commit.

## 2026-04-19

- Tightened the review prompt around evidence discipline: low AXIOM grades now require an explicit concrete evidence type, test-result claims are forbidden without listed tests, and final JSON is kept compact to reduce truncation.
- Made supervised review rewards more aggressive against functional-boundary mistakes and low-grade predictions without concrete evidence.
- Added a stronger no-test calibration rule: without a traced listed-test failure, grade 0/1 is reserved for unrelated code, certain runtime/syntax errors, or direct requirement contradictions; plausible but unproven defects should stay at 2/3.
- Adjusted AXIOM target construction for samples with executable tests: test-pass evidence now overrides contradictory CodeCritic correctness boundaries before mapping to AXIOM grades.
- Further constrained review evidence: listed-test failures must quote the exact assertion expectation, and equivalent mathematical transformations must not be treated as unrelated code solely due to naming/decomposition differences.
- Added executable evidence verification for `provided_test_failure`: if listed tests are absent or all pass, the evidence is marked false and receives a strong reward cap.
- Lowered the reward cap for unsupported `provided_test_failure` evidence further so MCTS has stronger pressure to avoid fabricated test-failure leaves.
- Added an AST-based `unused_identifier` verifier for Python snippets: claims that a quoted variable/parameter is unused are checked against actual identifier load/store/parameter usage and strongly capped when unsupported.
- Added optional verifier-correction training export in `preprocess_review_mcts_data.py`: verifier-rejected leaves can now produce explicit feedback-and-revision samples so LM fine-tuning can learn the correction pattern instead of seeing only abrupt value penalties.
- Improved verifier-correction feedback for executable claims by including the actual observed return value or exception when a claimed call result is unsupported.
- Added `--verifier_correction_repeat` to optionally oversample verifier-correction training items in short ablation runs without changing the default preprocessing behavior.
- Added a resumable verifier-correction ablation workflow script plus an evaluation summarizer for valid rate, AXIOM grade error, correctness-boundary accuracy, and unsupported evidence rates.
- Added a batch review evaluator that loads policy/value models once for multi-record evaluation, plus an overnight repeat2 verifier-correction ablation script that trains matched baseline/correction checkpoints on separate GPUs and evaluates balanced held-out CodeCritic samples.
- Added a queued verifier-correction repeat sweep script: after the repeat2 overnight run exits, it trains/evaluates repeat1 and repeat4 correction variants on the same held-out indices for oversampling-strength comparison.
- Added `filter_review_train_data.py` to pre-filter review training JSONL by the same Qwen review prompt token budget used by training, preventing most samples from being discarded inside `train_multi`.
- Added a filtered1536 verifier-correction ablation script that uses pre-filtered token-budgeted data, weak correction LM weight, no correction oversampling, and no-proxy ntfy notifications.
- Added a safer filtered1152 verifier-correction ablation script after filtered1536 OOMed on 4090 memory; it keeps token-budget filtering and weak correction mixing but lowers sequence length and enables expandable CUDA segments.
- Relaxed filtered1152 policy-count guards to match the observed 1152-token retention rate (baseline 24 policy items, correction roughly 42).
- Removed the review prompt instruction `do not default to high scores`.
- Added prompt calibration rules requiring concrete functional-defect evidence before assigning AXIOM grades 0-2, so correct-but-imperfect code stays within grades 3-5.
- Confirmed `supervised_medium_20260418` completed successfully but its ntfy notification was not received.
- Added retrying ntfy notifications and `notify.log` diagnostics to the supervised medium run script.

## 2026-04-18

- Added a long-term experiment plan and a standard AXIOM+CodeCritic supervised medium-run pipeline.
- Added unified review-scoring dataset preparation for AXIOMBench and CodeCriticBench.
- Updated review prompts to use the sample code language instead of hard-coding Python fences.
- Removed previously tracked Python bytecode/cache files from git tracking. Runtime caches are now covered by `.gitignore`.
- Removed base-model text self-judging for review MCTS node scoring.
- `ReviewMCTS` now rejects configs with `self_review_value_func: True` to prevent accidental self-labeled data generation.
- Kept objective reward labeling and future trained value-head inference paths.
- Updated score summarization so predictions are no longer parsed from `self_judge`.
- Added ignore rules for Python bytecode/cache files to avoid committing runtime artifacts.
