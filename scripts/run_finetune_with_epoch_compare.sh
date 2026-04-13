#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <run_name> <manifest> <compare_input_clip> <compare_output_root> [extra finetune args...]" >&2
  exit 1
fi

run_name="$1"
manifest="$2"
compare_input_clip="$3"
compare_output_root="$4"
shift 4

timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="models/deoldify/finetune/${run_name}"
epoch_dir="${output_dir}/epoch_checkpoints"
log_dir="data/logs/finetune"
log_path="${log_dir}/${run_name}_${timestamp}.log"

mkdir -p "${log_dir}" "${output_dir}"

echo "output_dir=${output_dir}"
echo "epoch_dir=${epoch_dir}"
echo "log_path=${log_path}"
echo "compare_output_root=${compare_output_root}"

{
  caffeinate -dimsu uv run python scripts/finetune_deoldify.py \
    --manifest "${manifest}" \
    --output-dir "${output_dir}" \
    --resume \
    --save-every-epoch \
    "$@"

  uv run python scripts/render_epoch_comparisons.py \
    --input-clip "${compare_input_clip}" \
    --epoch-dir "${epoch_dir}" \
    --output-root "${compare_output_root}" \
    --overwrite
} 2>&1 | tee -a "${log_path}"
