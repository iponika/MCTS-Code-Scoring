# Codex Change Log

This file records Codex-made project changes so work can be resumed safely across forks and sessions. Each future code or workflow change should update this log and be followed by a git commit.

## 2026-04-19

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
