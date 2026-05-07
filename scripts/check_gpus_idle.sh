#!/usr/bin/env bash
set -euo pipefail

non_xorg_processes="$(
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
)"

if [[ -n "${non_xorg_processes// }" ]]; then
  echo "GPU compute processes are already running; refusing to start a GPU job." >&2
  echo "${non_xorg_processes}" >&2
  exit 1
fi

nvidia-smi
