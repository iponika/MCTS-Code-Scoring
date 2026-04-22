# Codex Change Log

This file records Codex-made project changes so work can be resumed safely across forks and sessions. Each future code or workflow change should update this log and be followed by a git commit.

## 2026-04-22

- Added section 4.4 to the group-meeting training report documenting the clean-balanced cross-dataset scoring experiment, including motivation, data construction, sample counts, metrics, and limitations for non-project readers.

## 2026-04-21

- Revised the principle-generalization training workflow after the short-prompt negative result: static score dataset responses no longer teach source/label evidence strings, the workflow token-filters before sampling, balances exact AXIOM grades 0-5 after filtering, lowers LM imitation weights, and defaults to a fresh 600-step balanced-clean v2 overnight run.
- Reworked `docs/group_meeting_slides_20260422/` into a dense 7-slide report deck, one slide per main section of `group_meeting_training_report_20260422.md`, preserving the report text and tables visibly instead of summarizing into short visual cards.
- Added `docs/group_meeting_slides_20260422/`, an html-ppt based slide deck for the group meeting report with speaker notes and keyboard/presenter-mode support.
- Added `docs/group_meeting_training_report_20260422.md`, a group-meeting report outline focused on model-training progress, effective technical choices, current difficulties, evaluation design, and questions for discussion.
- Added a compact `review_prompt_mode=short` training prompt and revised the principle-generalization workflow to use batch size 1 with longer 1152-token context while disabling pairwise ranking for this absolute-score stabilization stage.
- Added `run_principle_generalization_eval.sh` for a distinct principle-generalization experiment: AXIOM/CodeCritic dominate the training mix, Code-DiTing and CodeJudgeBench are low-weight auxiliary labels, and outputs use a fresh run/checkpoint namespace to avoid reusing old loss-alignment results.
- Added batch-local pairwise value ranking loss for paired CodeJudgeBench labels, plus score-dataset preprocessing support for top-level pair metadata and a cross-dataset loss-alignment training/eval workflow; the workflow defaults to a shorter 640-token training length so batch-local pairs fit on a 4090.
- Added `run_cross_dataset_review_eval.sh` to build a combined CodeCritic/AXIOM/Code-DiTing/CodeJudgeBench held-out manifest and compare base direct, trained direct, and trained value-reranked final-only scoring in a resumable tmux-friendly workflow.
- Added a multi-dataset eval manifest builder for CodeCritic, AXIOM, Code-DiTing, and CodeJudgeBench, plus final-only review generation, lenient grade parsing, interval/pairwise summary metrics, value-guided final-candidate penalties, and an optional AXIOM boundary auxiliary value loss.
- Expanded the report pilot brief into an objective group-report draft covering supervised MCTS, verifier value-only ablations, report pilot training, score-key ablation, CodeCritic repeats, and the negative AXIOM held-out result.
- Fixed the AXIOM v3 eval file path and made `--skip_value_scoring --share_policy_value_model` load LoRA checkpoints through the existing value-wrapper path while disabling value scoring.
- Added `--skip_value_scoring` to review evaluators so direct baselines do not load or run value-head scoring, and revised the AXIOM held-out report eval to a fresh 30-item v3 run with tighter prompt truncation.
- Added review prompt truncation and revised the AXIOM held-out report eval to write a fresh `axiom_report_eval_v2_20260421` run with shorter generations and no final retry, avoiding OOM on long AXIOM tasks.
- Added direct AXIOM raw-record support to the review evaluator and updated review prompts to preserve the sample language in code fences.
- Added `run_axiom_report_eval.sh` to build a balanced 60-item AXIOM held-out set excluding AXIOM items used in the report checkpoint training mix, then compare base direct, trained direct, and trained value-guided mean inference.
- Changed the default value-guided MCTS candidate score from `last_value` to `response_mean_value`, and added optional `--seed` support to review evaluators for repeatable stochastic evaluation.
- Added `run_report_mean_mcts_repeat.sh` to run two additional seeded report-pilot repeats with `response_mean_value` and aggregate them with the existing mean-value run for presentation-ready stability statistics.
- Added `response_conservative_value` as a value-guided candidate selection score, defined as `0.7 * response_mean_value + 0.3 * response_min_value`, to penalize low-confidence spans without relying only on the final token value.
- Added `run_value_score_key_ablation.sh` to compare `response_mean_value` and `response_conservative_value` value-guided MCTS selection against the existing report-pilot checkpoint on the held-out CodeCritic indices.

## 2026-04-20

- Added `docs/report_pilot_20260421/` with a report-ready Chinese summary, a pure-Python SVG chart generator, and two SVG figures for the report pilot training results.
- Added `run_report_pilot_training.sh` to build a report-oriented static AXIOM/CodeCritic + repeated MCTS value-only training mix, train a 480-step LoRA/value-head checkpoint, and compare base direct, report direct, and report value-guided MCTS outputs.
- Added `run_verifier_valueonly_repeat3.sh` to run deterministic evaluation of the current value-only verifier ablation and three fresh seeded baseline-vs-value-only repeats with aggregate summaries.
- Added `run_verifier_valueonly_ablation.sh` for a clean filtered1152 baseline-vs-value-only verifier correction comparison, training both fresh checkpoints on separate GPUs and writing a post-eval comparison summary.
- Changed verifier-correction export to default to value-only supervision: synthetic corrections no longer produce LM loss unless `--verifier_correction_mode policy` or `paired_repair` is explicitly selected. Original MCTS leaves with verifier-flagged unsupported evidence are also forced out of policy imitation.

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
