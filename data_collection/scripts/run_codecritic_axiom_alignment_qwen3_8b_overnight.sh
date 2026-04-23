#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data1/xianzhiwei/mcts-code-review}"
MODEL_KEY="${MODEL_KEY:-Qwen/Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-/data1/xianzhiwei/model/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
RUN_NAME="${RUN_NAME:-codecritic_axiom_alignment_qwen3_8b_overnight_20260423}"
MAX_TRAINING_SEQ_LENGTH="${MAX_TRAINING_SEQ_LENGTH:-4096}"
MAX_STEPS="${MAX_STEPS:-900}"
EXACT_PER_GRADE="${EXACT_PER_GRADE:-220}"
DROP_GRADE_ZERO="${DROP_GRADE_ZERO:-1}"
AXIOM_PER_GRADE="${AXIOM_PER_GRADE:-20}"
NTFY_URL="${NTFY_URL:-https://ntfy.sh/iponika_mcts}"

RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
CANDIDATE_DATA="${ROOT}/model_training/review_mcts_train_data/${RUN_NAME}_candidates.jsonl"
TRAIN_DATA="${ROOT}/model_training/review_mcts_train_data/${RUN_NAME}.jsonl"
OUTPUT_MODEL="${ROOT}/model_training/src/output/review-lora-${RUN_NAME}-${MAX_STEPS}step"
AXIOM_EVAL_RUN="${RUN_NAME}_axiom_clean_eval"
AXIOM_EVAL_ROOT="${ROOT}/model_training/src/output/review-eval-${AXIOM_EVAL_RUN}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE="init"

notify() {
  local status="$1"
  local message="${RUN_NAME} ${status}: stage=${CURRENT_STAGE}, host=$(hostname), time=$(date -Is), log=${LOG_FILE}, output=${OUTPUT_MODEL}"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    curl --noproxy '*' --connect-timeout 5 --max-time 20 -fsS -d "${message}" "${NTFY_URL}" >/dev/null 2>&1 || true
}

on_exit() {
  local code=$?
  if [[ "${CURRENT_STAGE}" != "finished" ]]; then
    notify "FAILED(code=${code})"
  fi
  exit "${code}"
}
trap on_exit EXIT

prepare_data() {
  CURRENT_STAGE="prepare_data"
  cd "${ROOT}"
  if [[ ! -f "${CANDIDATE_DATA}" ]]; then
    PYTHONPATH="${ROOT}/model_training/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    HF_HUB_OFFLINE=1 \
    uv run python -m magicoder.preprocess_score_datasets \
      --codecriticbench "${ROOT}/datasets/CodeCriticBench/data/CodeCriticBench.jsonl" \
      --train_lm_exact \
      --limit_per_source 0 \
      --disable_shuffle \
      --output_file "${CANDIDATE_DATA}"
  fi

  PYTHONPATH="${ROOT}/model_training/src" \
  PYTHONDONTWRITEBYTECODE=1 \
  HF_HUB_OFFLINE=1 \
  UV_CACHE_DIR=/tmp/uv-cache \
  uv run python - <<PY
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from magicoder.prompt_template import QWEN_REVIEW_STEP_PROMPT

candidate_path = Path("${CANDIDATE_DATA}")
train_path = Path("${TRAIN_DATA}")
max_tokens = int("${MAX_TRAINING_SEQ_LENGTH}")
exact_per_grade = int("${EXACT_PER_GRADE}")
drop_grade_zero = "${DROP_GRADE_ZERO}".strip().lower() not in {"0", "false", "no", "off"}
rng = random.Random(20260423)

rows = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines() if line.strip()]
tokenizer = AutoTokenizer.from_pretrained("${MODEL_PATH}", trust_remote_code=True)

def normalize_response(row):
    response = row.get("response")
    if isinstance(response, list) and response:
        return str(response[0]).strip() + "\n"
    if response:
        return str(response).strip() + "\n"
    return ""

def token_length(row):
    prompt = QWEN_REVIEW_STEP_PROMPT.format(instruction=row.get("instruction", ""), response="")
    return len(tokenizer.encode(prompt, add_special_tokens=True)) + len(tokenizer.encode(normalize_response(row), add_special_tokens=False)) + 1

for row in rows:
    row["token_length"] = token_length(row)

fit_rows = [row for row in rows if row["token_length"] <= max_tokens]
too_long_rows = [row for row in rows if row["token_length"] > max_tokens]
if drop_grade_zero:
    fit_rows = [row for row in fit_rows if int(row.get("target_axiom_grade", -1)) != 0]

by_grade = defaultdict(list)
for row in fit_rows:
    grade = row.get("target_axiom_grade")
    if isinstance(grade, int):
        by_grade[grade].append(row)

selected = []
selected_grades = [grade for grade in sorted(by_grade) if not (drop_grade_zero and grade == 0)]
for grade in selected_grades:
    items = list(by_grade[grade])
    rng.shuffle(items)
    chosen = items[:exact_per_grade]
    for row in chosen:
        row["value_loss_weight"] = 0.9
        row["lm_loss_weight"] = 0.12 if row.get("train_lm") else 0.0
        row["codecritic_axiom_alignment"] = True
        row["alignment_split"] = "train"
    selected.extend(chosen)

rng.shuffle(selected)
train_path.parent.mkdir(parents=True, exist_ok=True)
train_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")

summary = {
    "candidate_items": len(rows),
    "fit_items": len(fit_rows),
    "too_long_items": len(too_long_rows),
    "drop_grade_zero": drop_grade_zero,
    "selected_items": len(selected),
    "selected_grades": selected_grades,
    "fit_grade_counts": dict(Counter(str(row.get("target_axiom_grade")) for row in fit_rows)),
    "selected_grade_counts": dict(Counter(str(row.get("target_axiom_grade")) for row in selected)),
    "max_training_seq_length": max_tokens,
    "exact_per_grade": exact_per_grade,
}
(Path("${RUN_DIR}") / "train_data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY
}

train_model() {
  CURRENT_STAGE="train_model"
  local log_file="${LOG_DIR}/train.log"
  if [[ -f "${OUTPUT_MODEL}/adapter_model.safetensors" && -f "${OUTPUT_MODEL}/value_head.pth" ]]; then
    echo "[stage:${CURRENT_STAGE}] final checkpoint exists, skipping: ${OUTPUT_MODEL}"
    return
  fi
  cd "${ROOT}/model_training/src"
  resume_args=()
  latest_checkpoint="$(find "${OUTPUT_MODEL}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${latest_checkpoint}" ]]; then
    resume_args=(--resume_from_checkpoint "${latest_checkpoint}")
    echo "[stage:${CURRENT_STAGE}] resuming from ${latest_checkpoint}"
  fi
  echo "[stage:${CURRENT_STAGE}] start=$(date -Is)"
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_DATASETS_CACHE=/tmp/hf-datasets-cache \
  HF_HUB_OFFLINE=1 \
  TRL_EXPERIMENTAL_SILENCE=1 \
  UV_CACHE_DIR=/tmp/uv-cache \
  uv run python -m magicoder.train_multi \
    --task review \
    --model_key "${MODEL_KEY}" \
    --model_name_or_path "${MODEL_PATH}" \
    --datafile_paths "../review_mcts_train_data/$(basename "${TRAIN_DATA}")" \
    --output_dir "${OUTPUT_MODEL}" \
    --max_steps "${MAX_STEPS}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_training_seq_length "${MAX_TRAINING_SEQ_LENGTH}" \
    --bf16 True \
    --logging_steps 20 \
    --save_strategy steps \
    --save_steps 150 \
    --save_total_limit 3 \
    --report_to none \
    --optim adafactor \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_steps 40 \
    --peft lora \
    --value_weight 0.05 \
    --boundary_value_weight 0.02 \
    --pairwise_value_weight 0.0 \
    --disable_train_shuffle True \
    --train_sampling_strategy sequential \
    --num_proc 1 \
    --seed 20260423 \
    "${resume_args[@]}" >"${log_file}" 2>&1
  echo "[stage:${CURRENT_STAGE}] done=$(date -Is)"
}

run_axiom_eval() {
  CURRENT_STAGE="axiom_clean_eval"
  if [[ -f "${AXIOM_EVAL_ROOT}/comparison.json" ]]; then
    echo "[stage:${CURRENT_STAGE}] comparison exists, skipping: ${AXIOM_EVAL_ROOT}/comparison.json"
    return
  fi
  RUN_NAME="${AXIOM_EVAL_RUN}" \
  BASE_MODEL_PATH="${MODEL_PATH}" \
  TRAINED_MODEL_PATH="${OUTPUT_MODEL}" \
  TRAIN_DATA="${TRAIN_DATA}" \
  PER_GRADE="${AXIOM_PER_GRADE}" \
  DROP_AXIOM_GRADE_ZERO=1 \
  MAX_PROBLEM_CHARS=12000 \
  MAX_CODE_CHARS=12000 \
  MAX_NEW_TOKENS=256 \
  FINAL_MAX_NEW_TOKENS=320 \
  NTFY_URL="${NTFY_URL}" \
  "${ROOT}/data_collection/scripts/run_axiom_clean_eval.sh"
}

main() {
  echo "[start] run=${RUN_NAME} time=$(date -Is) log=${LOG_FILE}"
  prepare_data
  if [[ "${STOP_AFTER_DATA:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_DATA=1 data=${TRAIN_DATA}"
    exit 0
  fi
  train_model
  run_axiom_eval
  CURRENT_STAGE="finished"
  notify "FINISHED"
  echo "[finished] model=${OUTPUT_MODEL} eval=${AXIOM_EVAL_ROOT} log=${LOG_FILE}"
}

main "$@"
