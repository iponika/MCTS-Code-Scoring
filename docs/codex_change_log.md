# Codex Change Log

This file records Codex-made project changes so work can be resumed safely across forks and sessions. Each future code or workflow change should update this log and be followed by a git commit.

## 2026-04-25

- Added semantic review deduplication for MCTS and training export: sibling final reviews with the same AXIOM grade, verdict, correctness flag, repair effort, and evidence type are skipped, and preprocessing removes same-parent semantic duplicates before building training items.
- Reworked review MCTS termination into bounded exploration plus linear frontier rollout: branching stops at `review_explore_depth`/node budgets, unfinished frontier leaves are then advanced one continuation at a time until a natural `<review>` or max-depth forced final, and non-review expansions no longer backpropagate neutral rewards that dilute q-values.
- Made review prompts deployment-aligned by hiding dataset tests from the model by default across MCTS generation, direct baselines, bootstrap export, training preprocessing, and value-guided evaluation; tests remain available for offline reward/label computation and can be exposed only with the new `show_tests_in_prompt` diagnostic switch.
- Switched the default review pipeline to correctness-only scoring: model-visible prompts no longer mention target review dimensions or require a `dimension` JSON field, CodeCritic samples expose only `Correctness Verification` by default, and the main local/API MCTS configs now explore one correctness branch.
- Tightened objective review reward labeling: `compute_review_reward` now records AXIOM grade distance and caps rewards for predictions at least two AXIOM levels away from the target, so same-boundary but severely over/under-scored reviews no longer receive high q-values.
- Added `tools/mcts_tree_viewer.html`, a standalone browser viewer for MCTS review sample JSON/JSONL files. It renders the `react` tree, colors nodes by `q_value`, labels review leaves with parsed AXIOM grades, and opens node text/reward details on click.
- Removed the forced AXIOM contradiction filtering added in the prior bootstrap-rebalance pass. The project will debug and fix q-value/reward labeling directly instead of dropping high/low disagreement samples externally.
- Kept the non-filtering rebalance diagnostics: `data_collection/rebalance_review_train_data.py` can still report score-delta buckets and stratify sampling by `dataset_index` or delta bucket, but it no longer deletes samples based on target/predicted grade disagreement.
- Updated `data_collection/scripts/run_bootstrap_comparison_qwen3_4b.sh` so bootstrap comparison runs no longer pass any forced contradiction-filtering flag.

## 2026-04-24

- Added a written bootstrap-comparison experiment spec and implementation plan under `docs/superpowers/specs/` and `docs/superpowers/plans/`, defining the fair 4B comparison between `Static-SFT`, `Direct-Bootstrap-SFT`, and `MCTS-Bootstrap-SFT`.
- Added `data_collection/configs/mcts_code_review_qwen3_4b.yaml` so local-vLLM direct and MCTS generation can target `Qwen/Qwen3-4B-Instruct-2507` without reusing the older 9B review config.
- Added `data_collection/prepare_codecritic_axiom_seedset.py`, which builds a balanced CodeCritic-only prepared review seed set using final prepared AXIOM grades instead of only raw score buckets.
- Added `data_collection/direct_bootstrap_review.py`, exporting multi-sample direct review rollouts as pseudo-`react` trees so the existing MCTS preprocessing path can be reused for a non-tree bootstrap baseline.
- Added `data_collection/prepare_static_review_train_data.py`, converting prepared review samples directly into exact-label `train_multi` JSONL for the static supervision baseline.
- Added `data_collection/scripts/run_bootstrap_comparison_qwen3_4b.sh`, a unified 4B workflow that prepares shared seed data, generates direct and MCTS bootstrap data, converts all three training sets, trains the three checkpoints, and evaluates them on AXIOM clean held-out with stop flags for smoke runs.
- Added `data_collection/rebalance_review_train_data.py` and updated the 4B comparison runner so direct-bootstrap and MCTS-bootstrap training files are explicitly rebalanced to the same policy/value target counts before training; the static baseline is repeated to the same total-example budget.

## 2026-04-23

- Added `docs/axiom_alignment_strategy_20260423.md`, documenting the current recommendation to treat AXIOM as the scoring anchor, use CodeCriticBench as the main mapped training source, and reserve AXIOM for external alignment checks instead of mixing everything blindly.
- Added `data_collection/scripts/run_codecritic_axiom_alignment_qwen3_8b_overnight.sh`, a dedicated Qwen3-8B overnight workflow that builds a balanced CodeCritic-only AXIOM-style train set, resumes single-card 4096-token LoRA/value-head training, and then runs AXIOM clean held-out evaluation automatically.
- Updated `docs/qwen3_8b_setup.md` with the new CodeCritic-to-AXIOM overnight alignment workflow so the 8B path no longer only documents the mixed-source principle script.

## 2026-04-22
- Added Qwen3-8B as a supported review model key, plus 8B-specific principle-generalization and vLLM serve helper scripts. The 8B workflow defaults to 4096-token full-context training to reduce long-sample filtering while keeping conservative single-card LoRA training settings for smoke validation.
- Added low-grade calibration instructions to review prompts so complete plausible code is not scored 0-2 without concrete defect evidence, and made the Qwen3-4B principle workflow AXIOM-heavy via `AXIOM_EXACT_FRACTION=0.7`.
- Added a clean AXIOM no-zero evaluation workflow with final-only JSON, longer prompt budgets, and code truncation notices outside code blocks so prompt shortening is not misread as a syntax defect.
- Added configurable review-evaluator prompt budgets and an option to avoid inserting truncation markers inside candidate code blocks during clean cross-dataset evaluation.
- Updated principle-generalization training data construction to drop AXIOM source grade-0 samples by default, while preserving CodeCritic grade-0 examples and making the behavior switchable via `DROP_AXIOM_GRADE_ZERO`.
- Removed the prior memory-saving short review training prompt path from the main principle-generalization workflow, restored full review-step prompts, re-enabled batch-local pairwise value ranking, and raised the Qwen3-4B workflow context default to 3072 tokens.
- Made principle-generalization data ordering explicitly align CodeJudgeBench pairs on batch-size-2 boundaries so pairwise value ranking remains active even in small smoke runs.
- Added a timeout to principle-generalization ntfy notifications so completed tmux jobs cannot hang indefinitely when the notification endpoint is unreachable.
- Added Qwen3-4B-Instruct-2507 as a supported small model for review training, including a 4B-specific principle-generalization workflow, a vLLM serve helper, and setup documentation.
- Hardened the Qwen3-4B vLLM serve helper to use `VLLM_HOST` instead of the generic `HOST` environment variable, avoiding conda host-triplet collisions during service startup.
- Documented the required localhost `NO_PROXY` override for vLLM API smoke tests when the shell has an `http_proxy` configured.
- Adjusted the group meeting slide layout after user review: removed the less important implementation-path block from slide 2, expanded slides 2/3/4/5 into larger, more readable layouts, and increased experiment-table font sizes.
- Updated the group meeting HTML slides to match the latest report wording, and changed sections 3-6 from multi-column cards to vertically stacked report blocks for denser group-sharing readability.
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
