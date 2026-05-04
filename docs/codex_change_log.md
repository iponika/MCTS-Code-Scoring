# Codex Change Log

This file records Codex-made project changes so work can be resumed safely across forks and sessions. Each future code or workflow change should update this log and be followed by a git commit.

## 2026-05-01

- Set DeepSeek-R1-Distill-Qwen-7B as the default model for current review-scoring workflows by switching the generic review-MCTS config and AXIOM eval base default away from Qwen.
- Added DeepSeek wrapper scripts for bootstrap comparison, direct review-vs-stepcount comparison, and direct stepwise diagnostics while keeping Qwen wrappers as explicit legacy/reproduction entries.
- Updated README and data-collection workflow docs so future routine experiments use DeepSeek defaults.
- Removed empty `<think>` wrappers from review-only training messages, added DeepSeek chat-template prefix fallback in `train_multi.py`, and added regression tests for the new message/render contract.
- Tightened DeepSeek direct-bootstrap final review formatting by prefilling the final `<review>` JSON prefix and using deterministic final decoding, so the model is pushed to continue a JSON object instead of drifting into prose.

## 2026-04-30

- Added a narrow final-review consistency rule to review prompts: final AXIOM scoring now tells the model to reconcile supported previous-step evidence with the final verdict, so a supported counterexample or trace cannot be silently contradicted by the final `<review>`.
- Mirrored the rule across data-generation MCTS prompts, direct local/API review prompts, review training templates, stepwise evaluation, and value-guided evaluation; added prompt-visibility regression checks.
- Fixed native-thinking review step generation after sample inspection: previous steps are now placed after `@@ Response` and rendered as assistant history under the chat template, thinking-mode step generation stops on `</think>`, and native `<think>` bodies are compacted to complete evidence sentences instead of storing max-token-cut fragments.
- Tightened step prompts only around evidence increments and continuation, without adding explicit "do not output Okay" style constraints; a 1-sample Qwen3-4B smoke confirmed no mid-sentence step truncation and no `The user.../previous step...` meta-prefix after the assistant-history fix.

## 2026-04-29

- Slimmed the tracked repository for server migration: removed original SEER code-generation data/assets, Open-R1 reproduction files, cached CodeJudgeBench Arrow files, legacy Magicoder code-generation preprocessing/training scripts, and stale ablation wrappers.
- Replaced the root README with the current code-scoring project layout, active workflows, dataset placement, uv install notes, and tmux usage.
- Rewrote `data_collection/README.md` and `model_training/src/README.md` to document only the maintained review-MCTS/direct-bootstrap/training/evaluation paths.
- Added `docs/server_migration.md` with clone, environment, dataset, model-cache, smoke-test, and tmux checklist for moving to a new GPU server.
- Made maintained shell wrappers derive `ROOT` from their script path by default and removed hard-coded local model paths from Qwen3.5 smoke scripts.
- Replaced the old non-portable `requirements.txt` with a review-pipeline dependency list compatible with uv-based installation.
- Added `README.zh-CN.md`, linked it from the English README, and documented Codex session migration in `docs/server_migration.md`.
- Added `Qwen/Qwen3-4B` thinking/no-thinking review-MCTS configs, a vLLM serve helper, and prompt parsing support that turns native `<think>` blocks into review `<step>` nodes.
- Retired the old non-thinking `Qwen/Qwen3-4B-Instruct-2507` defaults in maintained scripts/configs and removed its dedicated serve helper.
- Made Qwen3-4B maintained comparison scripts default to the native-thinking config and normalized the legacy `DIRECT_POLICY_RESPONSE_MODE=review` spelling to `final_review`.
- Added DeepSeek-R1-Distill-Qwen-7B review-MCTS config, vLLM serve helper, model-key support, and README smoke commands.
- Tightened review step prompts so prior `<step>` blocks are described as completed fixed context, non-final stepwise evaluation no longer advertises the final review JSON schema, and non-final generation explicitly forbids premature `<review>` output or repeated prior reasoning.
- Rewrote the maintained review prompts around correctness-only AXIOM scoring: removed SEER/code-generation persona noise from review paths, separated step-only/final-only prompt templates, and made review training choose a final-only prompt for review-only responses instead of always asking for stepwise reasoning.
- Removed unused legacy Magicoder few-shot/codegen prompt constants from `model_training/src/magicoder/prompt_template.py` while retaining the small compatibility templates still imported by non-review training branches.
- Removed the remaining unused `QWEN_STEP_PROMPT`/`DSC_PROMPT` compatibility templates from `model_training/src/magicoder/prompt_template.py` and made `train_multi.py` explicitly review-only to avoid accidental code-generation prompt fallback.
- Removed legacy code-generation prompt constants from `data_collection/mcts_math/prompts/prompt_sft.py`, retired the old `react_sft_prompt_wrap` path, and simplified local vLLM generation so review data collection cannot accidentally use code-generation direct-prompt mixing.
- Reworked maintained review prompts again after auditing the AXIOM paper wording: data generation now uses separate step/final templates instead of `{mode_instruction}` injection, step generation no longer has a 40-word cap, AXIOM grades use refinement-effort wording with localized "minor tweaking" vs structural "major refactoring" examples, and training/evaluation prompt wrappers pass the same AXIOM scale explicitly.
- Expanded the AXIOM operational rubric in prompts and grade descriptions using AXIOMBench perturbation-rule examples while marking all examples as illustrative rather than exhaustive criteria, so models treat examples as calibration anchors instead of hard scoring rules.

## 2026-04-28

- Added `run_qwen35_9b_direct_stepwise_vs_review_smoke.sh`, a minimal Qwen3.5-9B Direct-review-only vs Direct-stepwise comparison workflow using vLLM tensor-parallel generation, FSDP dual-4090 LoRA/value training, and small AXIOM stepwise evaluation.
- Added `--response_mode stepwise` and `--reasoning_steps` to `data_collection/direct_bootstrap_review.py`, enabling non-MCTS Direct-Bootstrap rollouts that sequentially generate fixed `<step>` reasoning blocks before the final `<review>`.
- Added `DIRECT_BOOTSTRAP_RESPONSE_MODE` and `DIRECT_BOOTSTRAP_REASONING_STEPS` environment controls to the 4B bootstrap comparison scripts so review-only Direct and stepwise Direct can be compared under the same training/evaluation workflow.
- Added `run_direct_stepwise_vs_review_qwen3_4b.sh`, a focused comparison workflow that reuses the latest review-only Direct baseline, trains only a new non-MCTS stepwise Direct checkpoint, and evaluates it with stepwise value-guided inference.
- Verified the new stepwise Direct react-tree shape with a no-GPU smoke test: a direct trajectory is exported as `c0 -> c0.0 -> c0.0.0 -> c0.0.0.0` and preprocesses into `early, early, middle, review` response segments.

## 2026-04-27

- Updated the 4B bootstrap-comparison workflow for the current stepwise-MCTS objective: default training context is now 3072 tokens, MCTS value-only paths are no longer downsampled to Direct by default, and optional `policy_response_mode=final_review` can collapse MCTS policy items to the final `<review>` for final-only scorer comparisons.
- Optimized review value-only batches in `RLTrainer`: when a batch has no LM-supervised tokens, Qwen-style models now use `logits_to_keep=1` and skip cross-entropy, avoiding unnecessary full sequence-by-vocabulary logits during value-head training.
- Added a `force_gradient_checkpointing` training switch and changed the Qwen3.5-9B FSDP smoke to use FSDP activation checkpointing instead of Trainer gradient checkpointing, avoiding the previous conflict while reducing backward activation memory.
- Parameterized LoRA rank/alpha/dropout/target scope for review training and set the Qwen3.5-9B FSDP smoke to a lighter rank-8 attention-only LoRA profile, preserving the previous rank-64 full-scope defaults for other scripts.
- Added a configurable `FSDP_OFFLOAD_PARAMS` switch to the Qwen3.5-9B FSDP smoke script after confirming the first real backward pass OOMs even on an 888-token batch without CPU offload.
- Revised the Qwen3.5-9B FSDP smoke launcher to keep Accelerate mixed precision off by default while loading/casting the model wrapper itself to bf16, avoiding Accelerate's FSDP fp32 flat-parameter upcast that OOMs on dual 4090s before training begins.
- Normalized the review value-head training wrapper to bf16 in non-inference mode so Accelerate FSDP can flatten Qwen3.5-9B decoder modules without mixed bf16/fp32 LoRA/value-head parameters.
- Added a Qwen3.5-9B dual-4090 FSDP smoke workflow and documentation. The new script launches `magicoder.train_multi` through Accelerate FSDP with `Qwen3_5DecoderLayer` wrapping, LoRA/value-head training, token-length auditing, resume support, and ntfy notification so 4096/6144/8192-token capacity can be probed without the old single-card truncation pressure.
- Added a policy-imitation quality gate in review MCTS preprocessing: exact-grade high-q paths with empty/`none` final evidence, premature JSON review steps, or near-duplicate reasoning steps are kept as value-only supervision instead of being imitated by the policy model.
- Tightened review-MCTS step generation after tree inspection: non-final prompts now hide the final `<review>` JSON schema, require each `<step>` to add new evidence or challenge a prior claim, and skip near-duplicate step children that merely restate a parent or sibling.
- Parameterized AXIOM clean evaluation so it can run either final-only scoring or stepwise review-MCTS inference, with configurable `SCORE_KEY`, step/candidate counts, rethink thresholds, and optional base/trained eval stages.
- Added a stepwise MCTS score-key comparison wrapper for the 2026-04-27 run: it reuses the latest MCTS checkpoint, disables `--final_only_json`, allows multi-step/multi-candidate value-reranked inference, compares `response_mean_value` against `last_value`, and writes a compact comparison against the existing Direct-Bootstrap final-only baseline.
- Switched review value-guided inference defaults back to `last_value` and hardened final `<review>` parsing: malformed review JSON with extra braces, unescaped control characters, or truncated evidence strings can now recover a minimal AXIOM grade payload when the grade is present, improving stepwise MCTS valid-rate without changing generated text.

## 2026-04-26

- Made CodeCritic AXIOM seed preparation observable and bounded: pass-rate evaluation now supports assertion caps/timeouts, the seed builder emits progress and elapsed metadata, and the 4B bootstrap comparison runner defaults to capped objective-test execution so first-stage seed preparation no longer appears stalled on larger overnight runs.
- Fixed CodeCritic seed objective-test handling so only Python `assert` tests are executed; stdin-style debug/Codeforces tests are now treated as non-executable metadata and no longer demote CodeCritic-correct samples to low AXIOM grades or cause long timeout-heavy seed preparation.
- Made the 4B bootstrap comparison runner's AXIOM held-out evaluation size configurable via `EVAL_PER_GRADE`, so overnight comparisons can trade runtime for more reliable cross-dataset metrics.
- Added an overnight Qwen3-4B stage-aware branch-depth-2 bootstrap comparison wrapper for the 2026-04-26 run, using 80 CodeCritic seed samples, 240 training steps per method, larger AXIOM held-out evaluation, compact summary export, and no-proxy ntfy notification.
- Fixed the overnight wrapper compact-summary step to respect overridden `RUN_NAME`, allowing clean reruns after aborted smoke/health-check attempts.
- Added stage-aware review value labels during MCTS training export: each path now preserves `raw_q_value`, records node tags/stage buckets/best descendant q, and defaults to blended value labels that combine raw tree q, best terminal descendant q, and stage-standardized q while keeping final `<review>` rewards direct.
- Fixed review MCTS tree shape for correctness-only runs: single-dimension review now starts directly from the root instead of creating an empty `0.0` dimension node, and branching exploration stops only after `review_explore_depth` rather than at it; local/API review configs now use `review_explore_depth: 2`.

## 2026-04-25

- Tightened review MCTS training export policy imitation: high-q best paths now require exact AXIOM grade agreement with the target to keep `train_lm=True`; one-grade over/under-scored paths remain as value-only supervision and are counted as `policy_grade_mismatch_paths`.
- Fixed `batch_review_evaluator.py` so the batch AXIOM evaluation path accepts the same `show_tests_in_prompt` option as the single-record evaluator; default batch evaluation still hides dataset tests from the model.
- Relaxed the unsupported `provided_test_failure` reward cap when the review's AXIOM grade and functional boundary are correct: bad evidence typing still caps reward, but no longer ranks a correct-score review below a functional-boundary mistake.
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

## 2026-04-30

- Added `data_collection/review_experiment_utils.py` with helpers for normalizing direct-stepwise step-count variants and aligning direct-review policy-sample count to the static baseline without discarding extra value-only samples.
- Added `tests/test_review_experiment_utils.py` to lock the new step-count and review-alignment behavior with focused unit coverage.
- Added `data_collection/scripts/run_direct_stepcount_vs_review_qwen3_4b.sh` to run a direct comparison among `static`, `direct-review`, `direct-stepwise-1step`, and `direct-stepwise-2step`.
- The new workflow fixes the experiment contract for this phase: `static` and `direct-review` are matched only on final `<review>` policy-sample count, while stepwise variants are allowed to keep their extra `<step>` supervision so we can test whether progressive reasoning itself helps.
- The new workflow records provenance explicitly in `summary.json`: training seeds come from `data_collection/prepare_codecritic_axiom_seedset.py` over `datasets/CodeCriticBench/data/CodeCriticBench.jsonl`, and supervised evaluation seeds come from AXIOM clean held-out sampling in `data_collection/scripts/run_axiom_clean_eval.sh`.

## 2026-05-01

- Inspected recent stepwise evaluation artifacts and confirmed that many final `<review>` blocks restate or refine evidence already introduced in preceding `<step>` blocks, so the pre-review reasoning path is sometimes genuinely used rather than being pure noise.
- Fixed `model_training/src/magicoder/preprocess_review_mcts_data.py` so a mixed terminal node containing `step/thinking prefix + <review>` is no longer collapsed to just the `<review>` body during preprocessing.
- Added `extract_response_segments()` to split mixed terminal text into separate reasoning segments plus the final `<review>`, stripping orphan `<think>` markers while preserving usable pre-review evidence.
- Updated path-to-training conversion so these extracted prefix segments inherit the terminal node's q-value metadata and enter stepwise training instead of being silently discarded.
- Added focused unit coverage in `tests/test_review_training_dedupe.py` for mixed terminal `step + review` extraction and export.
- Added reasoning-artifact extraction for both local text outputs and API structured outputs: `extract_reasoning_artifacts()` now separates visible reasoning from final content using either `reasoning/reasoning_content` fields or fallback `<think>...</think>` parsing.
- Updated direct review/bootstrap exporters to store `reasoning` and `reasoning_source` alongside cleaned final review text, instead of leaving orphan `</think>` markers mixed into scored review payloads.
- Extended the OpenAI-compatible API generator so completion objects preserve structured `reasoning` metadata when the backend returns it.
- Updated `model_training/src/magicoder/review_evaluator.py` to record per-candidate reasoning metadata and a composed `final_reasoning` field in evaluation artifacts.
- Added focused parser tests for structured-reasoning preference and `<think>`-block fallback extraction in `tests/test_review_evaluator_parsing.py`.
- Added Qwen-official message-format training export: review training items now carry `messages` with a user turn and a single assistant turn whose content is `<think>` step evidence followed by the final `<review>`.
- Added `assistant_parts` metadata so each step/review part keeps its `q_value`, `q_min`, and `q_max` without exposing those labels to the model text.
- Updated `train_multi.py` to prefer `messages + assistant_parts`, render them through the tokenizer chat template, mask non-assistant tokens for LM loss, and place value labels at the corresponding assistant part boundaries.
- Updated static exact-label data, AXIOM/CodeCritic score preprocessing, and token-budget filtering to emit or consume the same Qwen message format.
- Added `tests/test_train_multi_qwen_messages.py` plus expanded preprocessing tests to verify chat-template masking, value-only behavior, and `assistant_parts` synchronization.
- Fixed `data_collection/direct_bootstrap_review.py` so direct final-review repeats no longer use `n>1` under greedy decoding. Review-only repeats are now expanded into duplicated prompts with `n=1`, which preserves multiple rollouts while staying compatible with vLLM's greedy-sampling constraints.
- Added `tests/test_direct_bootstrap_review.py` to lock the direct-review repeat behavior and prevent regressions where `temperature=0` is paired with `n>1`.
- Extended `train_multi.py`'s DeepSeek/Qwen chat-prefix compatibility: when neither the generation-prompt render nor the empty-assistant render is a prefix of the full assistant turn, training now falls back to the rendered conversation history before the assistant turn. This fixes DeepSeek review samples whose assistant message starts directly with `<review>` instead of a `<think>` prelude.
- Added a new regression case in `tests/test_train_multi_qwen_messages.py` covering the real DeepSeek pattern where only the user-history render is prefix-aligned.
- Tightened final review prompting for direct bootstrap: the response prefix now starts from `{"axiom_grade": ` instead of a bare `{`, and the final prompt explicitly requires the next value to be an integer `0-5` followed by a comma, with no `<think>`, `<step>`, markdown fence, or prose in the final turn.
- Added prompt-visibility assertions for the stricter final-review contract so future prompt edits keep the JSON-entry behavior locked.

## 2026-05-02

- Changed stepwise final-review prompting in `model_training/src/magicoder/review_evaluator.py` so previously completed `<step>` blocks are no longer injected as assistant-prefix text after `@@ Response`.
- Final stepwise reviews now receive prior `<step>` blocks as explicit evidence context inside the instruction body, while the final assistant response starts empty and must begin the `<review>` block itself.
- Kept non-final step prompting unchanged: intermediate step generation still continues from the existing assistant-side `<step>` history.
- Updated `tests/test_review_prompt_visibility.py` to lock the new split between stepwise-final prompting and stepwise-intermediate prompting.
- On branch `direct-final-free-review`, relaxed direct `review`-mode final prompting so it no longer pre-fills `<review>` / JSON or appends `/no_think`; the model may reason first and then finish with a final `<review>` block.
- Added a direct-review normalization path that keeps only the last complete `<review>...</review>` block from freeform outputs, so pre-review reasoning does not leak into exported policy text.
- Added focused unit coverage in `tests/test_direct_bootstrap_review.py` for the freeform direct-final prompt contract and last-review extraction behavior.
- On branch `direct-final-free-review`, changed data-generation `stepwise -> final review` prompting so accumulated `<step>` blocks move into instruction-side evidence context instead of staying in `@@ Response` as assistant-prefix text.
- The data-generation stepwise final prompt now keeps `@@ Response` for only the final `<review>` opening / JSON prefix, matching the evaluator-side structure and reducing continuation pressure from prior `<step>` text.
- Added prompt-visibility coverage to lock the new data-generation stepwise-final placement behavior.
- Backed up the pre-rewrite final-review prompts in `docs/prompt_backups/review_final_prompts_before_rewrite_20260502.md`.
- Rewrote data-generation and evaluator final-review prompts around functional-correctness AXIOM scoring: the task/code/tests appear before previous analysis notes, the final review synthesizes prior steps from instruction-side context, and the required JSON keeps `evidence_type` while dropping legacy `score` and `verdict` fields.
- Shortened the evidence rules so they preserve the core constraints, no unsupported test claims, low grades need concrete functional defects, 1-2 grounded evidence strings, without the older bulky evidence-system wording.
- Updated reward parsing to derive the legacy verdict alignment from `axiom_grade` when a new-format review omits `verdict`.
- Backed up the pre-rewrite step prompts in `docs/prompt_backups/review_step_prompts_before_rewrite_20260502.md`.
- Rewrote data-generation, training, and stepwise-evaluation step prompts to match the new final-review prompt style: step outputs are now JSON objects wrapped in `<step>` tags with `step_type`, `evidence_type`, `claim`, and `functional_implication`.
- Updated direct/bootstrap and value-guided stepwise prompts so previous steps are described as `Previous analysis notes` when placed in the instruction body, while assistant-prefix mode still treats already emitted steps as response history.
- Replaced new intermediate `<step>` generation with native reasoning notes to avoid nested `<think>` / `<step>` ambiguity in thinking models.
- Updated direct bootstrap, review MCTS parsing, stepwise evaluators, and Qwen-message training export so intermediate nodes are stored as plain reasoning text inside `<think>`, while old `<step>...</step>` samples remain readable as legacy input.
- Removed `</step>` from thinking-model stop lists; intermediate generation now stops on `</think>` so one model call corresponds to one native reasoning node.
- Fixed the multi-record evaluator CLI contract after the native-reasoning prompt rewrite: `model_training/src/magicoder/batch_review_evaluator.py` now exposes `--step_context_mode` with the same default (`instruction_context`) expected by `review_evaluator.evaluate_dimension()`.
- Added `tests/test_batch_review_evaluator.py` so future evaluator launches fail in unit tests instead of during long heldout runs when prompt/eval arguments drift out of sync.
- Diagnosed the interrupted long run `native_reasoning_prompt_long_20260502`: training completed for all four checkpoints, but the AXIOM eval stage aborted before writing `summary.json` because the batch evaluator parser lacked `step_context_mode`; the run was resumed after the parser fix.
- Added `docs/session_handoff_20260502.md` as a focused handoff brief for another Codex session. It summarizes the current default model, datasets, prompt/eval contracts, maintained wrappers, branch state, and known active risks so cleanup/refactor work can start from the actual mainline instead of older legacy scripts.

## 2026-05-03

- Diagnosed 100% `missing_review_tags` evaluation failure: root cause is a combination of training-data quality issues, `enable_thinking` mismatch, and unconstrained freeform data generation.
- Fixed `enable_thinking` mismatch in `model_training/src/magicoder/review_evaluator.py`: `build_chat_eval_prompt` now passes `enable_thinking=None` (tokenizer default) instead of `False`, matching the training-time `render_chat_template` which also omits this parameter.
- Added `_try_extract_review_json()` in `model_training/src/magicoder/preprocess_review_mcts_data.py` to salvage review JSON embedded in reasoning-only segments (e.g. inside markdown code fences or bare JSON). Rescued 22 direct-review and 68 stepwise-1step training samples that previously lost their structured review.
- Added an `lm_loss_weight` guard in `attach_qwen_messages()`: training items whose final assistant content lacks a `<review>` block are forced to `lm_loss_weight=0.0`, preventing the model from learning reasoning-only output without structured scoring. Neutralized 170 harmful direct-review training samples (previously 69% of the dataset with `lm_loss_weight > 0`).
- Changed `generate_review_only` in `data_collection/direct_bootstrap_review.py` to default `freeform_final_review=False`, so `FINAL_REVIEW_PREFILL` anchors the model output format. Added `--freeform_final_review` CLI flag to opt back into the old relaxed behavior. Non-freeform outputs now prepend `FINAL_REVIEW_PREFILL` before normalization, matching the `generate_stepwise` final-step pattern.

## 2026-05-04

- Fixed a chat-template evaluation regression introduced by the recent prompt refactor: `model_training/src/magicoder/review_evaluator.py` now builds chat-template prompts from the same `prompt_for_dimension()` raw-text contract used elsewhere, so `step_context_mode=instruction_context` and `step_context_mode=assistant_prefix` keep their intended semantics instead of silently collapsing to assistant-history continuation.
- Added prompt-visibility regression tests for both chat-template step-context modes in `tests/test_review_prompt_visibility.py`.
- Reverted the accidental `direct_review` default contract drift from the 2026-05-03 change: `data_collection/direct_bootstrap_review.py` again defaults to `freeform_final_review=True`, while `--no-freeform_final_review` explicitly enables the anchored `FINAL_REVIEW_PREFILL` path.
- Added focused unit coverage in `tests/test_direct_bootstrap_review.py` to lock both the freeform default and the explicit anchored override.
- Refactored review prompt construction into one shared interface in `shared/prompt_contract.py`: `build_review_prompt_from_sample()` now owns raw prompt construction for anchored/freeform final review, `instruction_context` vs `assistant_prefix`, parse-error retry prompts, and test visibility; `apply_review_prompt_controls()` now owns `/think` and `/no_think` suffix handling.
- Switched the main data-generation and evaluation callers (`data_collection/direct_bootstrap_review.py`, `data_collection/direct_review_local.py`, `data_collection/direct_review_api.py`, `data_collection/mcts_math/agents/utils.py`, and `model_training/src/magicoder/review_evaluator.py`) to the shared prompt builder so prompt changes no longer require duplicating edits across side-specific builders.
- Extended the shared sample-to-instruction path so data-generation style `tests_for_prompt` text can flow through the same builder as evaluator-side `tests` lists.
- Added shared-interface regression tests in `tests/test_prompt_contract_alignment.py` and updated prompt-visibility expectations to lock the new final-stage structure: previous analysis notes now live in instruction context by default for final review, while assistant-prefix mode is reserved for intermediate continuation only.
- Removed `data_collection/mcts_math/prompts/prompt_sft.py`. Data-collection-side callers now import the shared prompt contract directly with repo-root fallback, so `shared/prompt_contract.py` is the only prompt source and `model_training/src/magicoder/prompt_template.py` remains as the lone compatibility wrapper for older training-side imports.
- Moved the remaining training-side legacy prompt bodies (`QWEN_REVIEW_STEP_PROMPT`, `QWEN_REVIEW_STEP_ONLY_PROMPT`, `QWEN_REVIEW_FINAL_ONLY_PROMPT`) and `review_prompt_for_response()` into `shared/prompt_contract.py`. `model_training/src/magicoder/prompt_template.py` is now a thin alias/re-export shim instead of a second prompt-definition file.
- Temporary experiment for the `assistant_prefix` / chat-template intermediate-step path: when prior reasoning is present in assistant history, the prompt now tells the model that earlier analysis text may be inserted directly as prior thinking history, and asks it to first summarize the previously reached correctness status in one sentence before continuing the correctness judgment. This change is isolated to the assistant-history continuation mode and is intended to be easy to revert after comparison.
- 2026-05-04: Added `data_collection/configs/mcts_code_review_deepseek_r1_distill_qwen_7b_lowmem.yaml` for low-memory DeepSeek 7B experiment relaunches after vLLM startup failed at `gpu_memory_utilization=0.88` on dual 4090 host.
- 2026-05-04: Raised the default final-review evaluation token budget from `320`/`0->reuse max_new_tokens` to `768` along the main evaluation path (`run_axiom_clean_eval.sh`, `batch_review_evaluator.py`, `review_evaluator.py`) after repeated final-review truncation before `<review>` emission.
