#!/usr/bin/env bash
set -euo pipefail

# Run directly on a GPU VM; no Slurm scheduler is needed.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
python_executable="${1:-$PWD/.venv/bin/python}"
# Pass the image folder and a new output folder, such as /path/to/A100.
data_folder="${2:?Usage: bash run.sh PYTHON DATA_FOLDER OUTPUT_FOLDER}"
output_folder="${3:?Choose a new GPU output folder; existing results are never overwritten}"

# Refuse to overwrite a previous run, and keep errors with partial results.
mkdir -p "$(dirname "$output_folder")"
mkdir "$output_folder"
support_folder="$output_folder/.support"
mkdir "$support_folder"
echo "Experiment log: $support_folder/run.log"
exec > "$support_folder/run.log" 2>&1
trap 'status=$?; echo "Run finished with exit code $status at $(date -Is)"; exit "$status"' EXIT
echo "Starting measurements on $(hostname) at $(date -Is)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f main.py || ! -f roofline.py || ! -d "$data_folder" ]]; then
    echo "Run from the code folder and make sure $data_folder is available." >&2
    exit 1
fi

# Supply the actual GPU board's ordinary FP32 peak and DRAM bandwidth.
: "${PEAK_GFLOPS:?Set PEAK_GFLOPS for the selected GPU}"
: "${BANDWIDTH_GBS:?Set BANDWIDTH_GBS for the selected GPU}"
ncu_executable="${NCU:-ncu}"
command -v "$ncu_executable" >/dev/null || {
    echo "Nsight Compute is missing. Install ncu 2024.4+ or set NCU to its path." >&2
    exit 1
}
"$python_executable" -c 'import math, sys, matplotlib; assert all(math.isfinite(float(x)) and float(x) > 0 for x in sys.argv[1:])' "$PEAK_GFLOPS" "$BANDWIDTH_GBS"

# keep results and environment details together for each job
nvidia-smi > "$support_folder/gpu.txt"
"$ncu_executable" --version > "$support_folder/ncu-version.txt"
"$python_executable" -m pip freeze > "$support_folder/packages.txt"

# run the three models one at a time on the allocated gpu
for model in resnet50 resnet101 resnet152; do
    model_folder="$output_folder/ResNet-${model#resnet}"
    echo "Running $model"
    "$python_executable" -u main.py --data "$data_folder" --model "$model" --output "$model_folder"
done

# profile one warmed-up step per phase, separately from timing
metrics="smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum"
for model in resnet50 resnet101 resnet152; do
    model_folder="$output_folder/ResNet-${model#resnet}"
    model_support="$support_folder/ResNet-${model#resnet}"
    mkdir -p "$model_support"
    for phase in forward backward; do
        echo "Profiling $model $phase"
        "$ncu_executable" --nvtx --nvtx-push-pop-scope process --nvtx-include "$phase/" \
            --metrics "$metrics" --replay-mode kernel --clock-control none \
            --cache-control all --csv --page raw --print-units base \
            --log-file "$model_folder/$model-$phase.csv" \
            --export "$model_support/$model-$phase" \
            "$python_executable" -u main.py --data "$data_folder" --model "$model" \
            --output "$model_support/$phase" --profile
        "$python_executable" -c 'import sys; from pathlib import Path; from roofline import read_kernels; read_kernels(Path(sys.argv[1]))' "$model_folder/$model-$phase.csv"
    done
done

"$python_executable" roofline.py "$output_folder" \
    --peak-gflops "$PEAK_GFLOPS" --bandwidth-gbs "$BANDWIDTH_GBS"
echo "Timing, profiles, and roofline plots saved in $output_folder"
