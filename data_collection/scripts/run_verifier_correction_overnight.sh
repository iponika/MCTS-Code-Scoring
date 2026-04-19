#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/xianzhiwei/mcts-code-review"
RUN_NAME="verifier_correction_overnight_20260419"
MODEL_PATH="/data1/xianzhiwei/model/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"

RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
EVAL_ROOT="${ROOT}/model_training/src/output/review-eval-${RUN_NAME}"
INDICES_FILE="${RUN_DIR}/heldout_indices.json"
BASE_DATA="${ROOT}/model_training/review_mcts_train_data/verifier_overnight_baseline_20260419.jsonl"
CORR_DATA="${ROOT}/model_training/review_mcts_train_data/verifier_overnight_correction_repeat2_20260419.jsonl"
BASE_OUT="${ROOT}/model_training/src/output/review-lora-verifier-overnight-baseline-120step"
CORR_OUT="${ROOT}/model_training/src/output/review-lora-verifier-overnight-correction-repeat2-120step"

INPUTS=(
  "${ROOT}/data_collection/review_mcts_runs/evidence_style_smoke4_v3_20260419/samples"
  "${ROOT}/data_collection/review_mcts_runs/supervised_medium_20260418/samples"
)

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${EVAL_ROOT}"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

NOTIFY_URL="${NTFY_URL:-https://ntfy.sh/iponika_mcts}"
NOTIFY_RESPONSE="${RUN_DIR}/notify_response.json"
NOTIFY_ERROR="${RUN_DIR}/notify_error.log"
CURRENT_STAGE="init"

notify() {
  local status="$1"
  local message="${RUN_NAME} ${status}: stage=${CURRENT_STAGE}, host=$(hostname), time=$(date -Is), log=${LOG_FILE}"
  if curl -fsS -d "${message}" "${NOTIFY_URL}" >"${NOTIFY_RESPONSE}" 2>"${NOTIFY_ERROR}"; then
    echo "[notify] sent ${status}"
  else
    echo "[notify] failed ${status}; see ${NOTIFY_ERROR}" >&2
  fi
}

on_exit() {
  local code=$?
  if [[ "${CURRENT_STAGE}" != "finished" ]]; then
    notify "FAILED(code=${code})"
  fi
  exit "${code}"
}
trap on_exit EXIT

select_indices() {
  CURRENT_STAGE="select_indices"
  if [[ -f "${INDICES_FILE}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${INDICES_FILE}"
    cat "${INDICES_FILE}"
    return
  fi
  cd "${ROOT}"
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${ROOT}/model_training/src" \
  uv run python - <<'PY'
import json
import random
from collections import defaultdict
from pathlib import Path

from magicoder.axiom_scoring import axiom_grade_from_codecritic

root = Path("/data1/xianzhiwei/mcts-code-review")
used = set()
for sample_path in (root / "data_collection/review_mcts_runs/supervised_medium_20260418/samples").glob("*.json"):
    try:
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if sample.get("source") == "mbpp" and isinstance(sample.get("dataset_index"), int):
        used.add(sample["dataset_index"])
used.update({53, 54, 55, 58, 59, 60})

by_grade = defaultdict(list)
dataset = root / "datasets/CodeCriticBench/data/CodeCriticBench.jsonl"
with dataset.open("r", encoding="utf-8") as handle:
    for idx, line in enumerate(handle):
        if idx in used:
            continue
        row = json.loads(line)
        if row.get("source") != "mbpp":
            continue
        grade = axiom_grade_from_codecritic(row.get("correctness"), row.get("score"))
        if grade is None:
            continue
        by_grade[int(grade)].append(idx)

rng = random.Random(20260419)
indices = []
grade_dist = {}
for grade in range(6):
    candidates = list(by_grade[grade])
    rng.shuffle(candidates)
    selected = sorted(candidates[:8])
    indices.extend(selected)
    grade_dist[str(grade)] = selected
payload = {"indices": indices, "per_grade": 8, "grade_dist": grade_dist, "excluded_count": len(used)}
output = root / "data_collection/review_mcts_runs/verifier_correction_overnight_20260419/heldout_indices.json"
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

convert_data() {
  CURRENT_STAGE="convert"
  cd "${ROOT}"
  if [[ ! -f "${BASE_DATA}" ]]; then
    echo "[stage:${CURRENT_STAGE}] baseline start=$(date -Is)"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${ROOT}/model_training/src" \
    HF_HUB_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.preprocess_review_mcts_data \
      --input "${INPUTS[@]}" \
      --output_file "${BASE_DATA}" \
      --policy_min_q 0.5 \
      --max_value_paths_per_dimension 0 \
      --shuffle_seed 20260419
  else
    echo "[stage:${CURRENT_STAGE}] baseline exists: ${BASE_DATA}"
  fi

  if [[ ! -f "${CORR_DATA}" ]]; then
    echo "[stage:${CURRENT_STAGE}] correction repeat2 start=$(date -Is)"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${ROOT}/model_training/src" \
    HF_HUB_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.preprocess_review_mcts_data \
      --input "${INPUTS[@]}" \
      --output_file "${CORR_DATA}" \
      --policy_min_q 0.5 \
      --max_value_paths_per_dimension 0 \
      --emit_verifier_corrections \
      --verifier_correction_repeat 2 \
      --verifier_correction_q 0.8 \
      --verifier_correction_lm_weight 0.5 \
      --verifier_correction_value_weight 0.5 \
      --shuffle_seed 20260419
  else
    echo "[stage:${CURRENT_STAGE}] correction exists: ${CORR_DATA}"
  fi
  wc -l "${BASE_DATA}" "${CORR_DATA}"
}

train_one() {
  local tag="$1"
  local cuda_device="$2"
  local data_file="$3"
  local out_dir="$4"
  local log_file="${LOG_DIR}/train_${tag}.log"

  if [[ -f "${out_dir}/adapter_model.safetensors" && -f "${out_dir}/value_head.pth" ]]; then
    echo "[stage:train_${tag}] final checkpoint exists, skipping: ${out_dir}"
    return
  fi

  (
    set -euo pipefail
    cd "${ROOT}/model_training/src"
    RESUME_ARGS=()
    LATEST_CHECKPOINT="$(find "${out_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "${LATEST_CHECKPOINT}" ]]; then
      RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CHECKPOINT}")
      echo "[stage:train_${tag}] resuming from ${LATEST_CHECKPOINT}"
    fi
    echo "[stage:train_${tag}] start=$(date -Is) cuda=${cuda_device}"
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_DATASETS_CACHE=/tmp/hf-datasets-cache \
    HF_HUB_OFFLINE=1 \
    TRL_EXPERIMENTAL_SILENCE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.train_multi \
      --task review \
      --model_key Qwen/Qwen3.5-9B \
      --model_name_or_path "${MODEL_PATH}" \
      --datafile_paths "../review_mcts_train_data/$(basename "${data_file}")" \
      --output_dir "${out_dir}" \
      --max_steps 120 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --max_training_seq_length 1024 \
      --bf16 True \
      --logging_steps 10 \
      --save_strategy steps \
      --save_steps 60 \
      --save_total_limit 2 \
      --report_to none \
      --optim adafactor \
      --learning_rate 5e-5 \
      --lr_scheduler_type linear \
      --warmup_steps 12 \
      --peft lora \
      --value_weight 0.025 \
      --num_proc 1 \
      "${RESUME_ARGS[@]}"
    echo "[stage:train_${tag}] done=$(date -Is)"
  ) >"${log_file}" 2>&1 &
}

wait_for_train() {
  CURRENT_STAGE="train_parallel"
  train_one "baseline" 0 "${BASE_DATA}" "${BASE_OUT}"
  pid_base=$!
  train_one "correction_repeat2" 1 "${CORR_DATA}" "${CORR_OUT}"
  pid_corr=$!
  echo "[stage:${CURRENT_STAGE}] waiting baseline pid=${pid_base}, correction pid=${pid_corr}"
  wait "${pid_base}"
  wait "${pid_corr}"
  echo "[stage:${CURRENT_STAGE}] done=$(date -Is)"
}

eval_one() {
  local tag="$1"
  local cuda_device="$2"
  local out_dir="$3"
  local eval_dir="${EVAL_ROOT}/${tag}"
  local log_file="${LOG_DIR}/eval_${tag}.log"
  mkdir -p "${eval_dir}"

  (
    set -euo pipefail
    cd "${ROOT}/model_training/src"
    echo "[stage:eval_${tag}] start=$(date -Is) cuda=${cuda_device}"
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_DATASETS_CACHE=/tmp/hf-datasets-cache \
    HF_HUB_OFFLINE=1 \
    TRL_EXPERIMENTAL_SILENCE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.batch_review_evaluator \
      --policy_model_path "${out_dir}" \
      --value_model_path "${out_dir}" \
      --share_policy_value_model \
      --input_record ../../datasets/CodeCriticBench/data/CodeCriticBench.jsonl \
      --record_indices_file "${INDICES_FILE}" \
      --output_dir "${eval_dir}" \
      --dimensions "Correctness Verification" \
      --device cuda \
      --dtype bf16 \
      --max_steps 3 \
      --num_candidates 1 \
      --max_new_tokens 256 \
      --final_max_new_tokens 384 \
      --temperature 0.7 \
      --top_p 0.95 \
      --rethink_threshold -0.2 \
      --max_rethinks 1 \
      --max_final_retries 2
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${ROOT}/model_training/src:${ROOT}/data_collection" \
    HF_HUB_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python "${ROOT}/data_collection/scripts/summarize_review_eval_outputs.py" \
      --eval_dir "${eval_dir}" \
      --output "${eval_dir}/summary.json"
    echo "[stage:eval_${tag}] done=$(date -Is)"
  ) >"${log_file}" 2>&1 &
}

wait_for_eval() {
  CURRENT_STAGE="eval_parallel"
  eval_one "baseline" 0 "${BASE_OUT}"
  pid_base=$!
  eval_one "correction_repeat2" 1 "${CORR_OUT}"
  pid_corr=$!
  echo "[stage:${CURRENT_STAGE}] waiting baseline pid=${pid_base}, correction pid=${pid_corr}"
  wait "${pid_base}"
  wait "${pid_corr}"
  echo "[stage:${CURRENT_STAGE}] done=$(date -Is)"
}

echo "[pipeline] start=$(date -Is)"
echo "[pipeline] log=${LOG_FILE}"
select_indices
convert_data
wait_for_train
wait_for_eval

CURRENT_STAGE="finished"
echo "[pipeline] finished=$(date -Is)"
notify "COMPLETED"
