# Bootstrap Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fair 4B experiment pipeline that compares Static-SFT, Direct-Bootstrap-SFT, and MCTS-Bootstrap-SFT for AXIOM-aligned code scoring.

**Architecture:** Reuse the existing review dataset loader, reward function, MCTS preprocessing, training, and evaluation scripts. Add only the missing bridges: a balanced CodeCritic seed-set builder, a direct-bootstrap pseudo-tree exporter, a static exact-label converter, and one orchestration shell script.

**Tech Stack:** Python, shell, local vLLM, Transformers LoRA training, existing `magicoder`/`mcts_math` utilities.

---

### Task 1: Add Experiment Spec and Config Paths

**Files:**
- Create: `docs/superpowers/specs/2026-04-24-bootstrap-comparison-design.md`
- Create: `data_collection/configs/mcts_code_review_qwen3_4b.yaml`
- Modify: `docs/codex_change_log.md`

- [ ] **Step 1: Add the 4B review generation config**

Use the existing review-MCTS config shape, but set the model to `Qwen/Qwen3-4B-Instruct-2507` and keep generation conservative.

- [ ] **Step 2: Record the experiment intent in the changelog**

Add a short bullet describing the new 4B bootstrap-comparison workflow and the three-way baseline structure.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-04-24-bootstrap-comparison-design.md data_collection/configs/mcts_code_review_qwen3_4b.yaml docs/codex_change_log.md
git commit -m "docs: define bootstrap comparison experiment"
```

### Task 2: Build a Balanced CodeCritic AXIOM Seed Set

**Files:**
- Create: `data_collection/prepare_codecritic_axiom_seedset.py`

- [ ] **Step 1: Implement balanced per-grade seed selection**

Read `CodeCriticBench.jsonl`, map each sample to AXIOM grade, filter unusable rows, and write normalized prepared review samples with equal counts per grade.

- [ ] **Step 2: Emit metadata**

Write per-grade counts and selection parameters so later runs can prove both bootstrap methods used the same seeds.

- [ ] **Step 3: Run a small smoke command**

Run:

```bash
PYTHONPATH=data_collection UV_CACHE_DIR=/tmp/uv-cache uv run python data_collection/prepare_codecritic_axiom_seedset.py --output /tmp/codecritic_seed_smoke.jsonl --metadata /tmp/codecritic_seed_smoke.metadata.json --per_grade 2 --min_grade 1
```

Expected: output JSONL and metadata exist, each selected grade count is `2`.

- [ ] **Step 4: Commit**

```bash
git add data_collection/prepare_codecritic_axiom_seedset.py
git commit -m "feat: add codecritic axiom seedset builder"
```

### Task 3: Add Direct-Bootstrap Pseudo-Tree Export

**Files:**
- Create: `data_collection/direct_bootstrap_review.py`

- [ ] **Step 1: Reuse existing prompt and reward logic**

Load prepared review samples with `load_codecriticbench_dataset`, sample `K` independent final reviews per sample, and score each candidate with `compute_review_reward`.

- [ ] **Step 2: Export pseudo-`react` records**

For each candidate, create one terminal-only node with:

- `text`
- `q_value`
- `value`
- `target_dimension`
- `final_answer`
- `reward_details`

Also emit `best_reviews_by_dimension` so `preprocess_review_mcts_data.py` can mark policy paths.

- [ ] **Step 3: Run a smoke command**

Run on 2 samples with `repeats=2` and verify the output contains `react` and terminal nodes.

- [ ] **Step 4: Commit**

```bash
git add data_collection/direct_bootstrap_review.py
git commit -m "feat: add direct bootstrap review exporter"
```

### Task 4: Add Static Exact-Label Converter

**Files:**
- Create: `data_collection/prepare_static_review_train_data.py`

- [ ] **Step 1: Convert prepared review samples into `train_multi` JSONL**

Use the same review instruction style and AXIOM value scale as the bootstrap training data. Each seed sample should become one exact-label training item.

- [ ] **Step 2: Run a smoke command**

Run on the seed-set smoke file and verify it writes valid `instruction/response/q_value/train_lm` items.

- [ ] **Step 3: Commit**

```bash
git add data_collection/prepare_static_review_train_data.py
git commit -m "feat: add static review train data converter"
```

### Task 5: Add the Unified 4B Experiment Runner

**Files:**
- Create: `data_collection/scripts/run_bootstrap_comparison_qwen3_4b.sh`

- [ ] **Step 1: Prepare shared seed data**

The script should build one balanced CodeCritic seed set and reuse it for all groups.

- [ ] **Step 2: Generate bootstrap data**

Run:

- direct bootstrap exporter
- existing `solver_review.py` MCTS exporter

Both should write raw JSONL under the same run directory.

- [ ] **Step 3: Convert train data**

Run:

- static exact-label converter
- `preprocess_review_mcts_data.py` for direct bootstrap
- `preprocess_review_mcts_data.py` for MCTS bootstrap

- [ ] **Step 4: Train all three checkpoints**

Use the same 4B LoRA settings for:

- static
- direct-bootstrap
- mcts-bootstrap

- [ ] **Step 5: Evaluate on AXIOM clean held-out**

Call `run_axiom_clean_eval.sh` once per trained checkpoint and collect the resulting `comparison.json` files into one summary.

- [ ] **Step 6: Add stop flags**

Support `STOP_AFTER_DATA=1` and `STOP_AFTER_TRAIN=1` so the workflow can be smoke-tested without a full overnight run.

- [ ] **Step 7: Run a data-only smoke test**

Run:

```bash
STOP_AFTER_DATA=1 data_collection/scripts/run_bootstrap_comparison_qwen3_4b.sh
```

Expected: seed set, direct raw, MCTS raw, and three train JSONL files exist.

- [ ] **Step 8: Commit**

```bash
git add data_collection/scripts/run_bootstrap_comparison_qwen3_4b.sh
git commit -m "feat: add qwen3 4b bootstrap comparison workflow"
```

### Task 6: Final Verification

**Files:**
- Modify: `docs/codex_change_log.md`

- [ ] **Step 1: Run syntax checks**

Run:

```bash
bash -n data_collection/scripts/run_bootstrap_comparison_qwen3_4b.sh
python -m py_compile data_collection/prepare_codecritic_axiom_seedset.py data_collection/direct_bootstrap_review.py data_collection/prepare_static_review_train_data.py
```

- [ ] **Step 2: Run the smoke workflow**

Run the data-only smoke command and inspect the output tree.

- [ ] **Step 3: Update changelog with the completed pipeline details**

- [ ] **Step 4: Commit**

```bash
git add docs/codex_change_log.md
git commit -m "chore: verify bootstrap comparison workflow"
```
