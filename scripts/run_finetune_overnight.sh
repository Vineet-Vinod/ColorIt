#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <run_name> <manifest> [extra finetune args...]" >&2
  exit 1
fi

run_name="$1"
manifest="$2"
shift 2

timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="models/deoldify/finetune/${run_name}"
log_dir="data/logs/finetune"
log_path="${log_dir}/${run_name}_${timestamp}.log"

mkdir -p "${log_dir}" "${output_dir}"

echo "output_dir=${output_dir}"
echo "log_path=${log_path}"

exec caffeinate -dimsu uv run python scripts/finetune_deoldify.py \
  --manifest "${manifest}" \
  --output-dir "${output_dir}" \
  --resume \
  "$@" 2>&1 | tee -a "${log_path}"
