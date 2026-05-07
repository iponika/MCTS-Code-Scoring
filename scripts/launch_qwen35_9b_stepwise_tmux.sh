#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-qwen35_9b_stepwise}"

"${ROOT}/scripts/check_gpus_idle.sh" >/dev/null

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" "
set -euo pipefail
cd '${ROOT}'
source '${ROOT}/scripts/qwen35_9b_env.sh'
'${ROOT}/scripts/check_gpus_idle.sh'
RUN_NAME=\"\${RUN_NAME:-qwen35_9b_direct_stepwise_server_$(date +%Y%m%d_%H%M%S)}\" \
SEED_PER_GRADE=\"\${SEED_PER_GRADE:-4}\" \
DIRECT_REPEATS=\"\${DIRECT_REPEATS:-2}\" \
MAX_STEPS=\"\${MAX_STEPS:-40}\" \
MAX_TRAINING_SEQ_LENGTH=\"\${MAX_TRAINING_SEQ_LENGTH:-2048}\" \
HF_HUB_OFFLINE=\"\${HF_HUB_OFFLINE:-0}\" \
bash data_collection/scripts/run_qwen35_9b_direct_stepwise_vs_review_smoke.sh
"

echo "started tmux session: ${SESSION}"
echo "attach with: tmux attach -t ${SESSION}"
