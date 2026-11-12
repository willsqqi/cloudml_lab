#!/bin/bash
#SBATCH --job-name=resnet-benchmark
#SBATCH --account=csci_ga_3033_085-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# submit from the code folder; use the activated environment or a supplied python path
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the code folder.}"
python_executable="${1:-python}"
data_folder="${2:-../data/imagenet128}"
output_folder="results/${SLURM_JOB_PARTITION}-${SLURM_JOB_ID}"

if [[ ! -f main.py || ! -f roofline.py || ! -d "$data_folder" ]]; then
    echo "Run from the code folder and make sure $data_folder is available." >&2
    exit 1
fi

# supply the exact GPU board's ordinary FP32 peak and DRAM bandwidth at submission
: "${PEAK_GFLOPS:?Set PEAK_GFLOPS for the selected GPU}"
: "${BANDWIDTH_GBS:?Set BANDWIDTH_GBS for the selected GPU}"
ncu_executable="${NCU:-ncu}"
command -v "$ncu_executable" >/dev/null
"$python_executable" -c 'import math, sys, matplotlib; assert all(math.isfinite(float(x)) and float(x) > 0 for x in sys.argv[1:])' "$PEAK_GFLOPS" "$BANDWIDTH_GBS"

# keep results and environment details together for each job
mkdir -p "$output_folder"
nvidia-smi > "$output_folder/gpu.txt"
"$ncu_executable" --version > "$output_folder/ncu-version.txt"
"$python_executable" -m pip freeze > "$output_folder/packages.txt"

# run the three models one at a time on the allocated gpu
for model in resnet50 resnet101 resnet152; do
    echo "Running $model"
    "$python_executable" -u main.py --data "$data_folder" --model "$model" --output "$output_folder"
done

# profile one warmed-up step per phase, separately from timing
metrics="smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum"
for model in resnet50 resnet101 resnet152; do
    for phase in forward backward; do
        echo "Profiling $model $phase"
        "$ncu_executable" --nvtx --nvtx-push-pop-scope process --nvtx-include "$phase/" \
            --metrics "$metrics" --replay-mode kernel --clock-control none \
            --cache-control all --csv --page raw --print-units base \
            --print-metric-name name --log-file "$output_folder/$model-$phase.csv" \
            --export "$output_folder/$model-$phase" \
            "$python_executable" -u main.py --data "$data_folder" --model "$model" \
            --output "$output_folder/profile-metadata/$phase" --profile
        "$python_executable" -c 'import sys; from pathlib import Path; from roofline import read_kernels; read_kernels(Path(sys.argv[1]))' "$output_folder/$model-$phase.csv"
    done
done

"$python_executable" roofline.py "$output_folder" \
    --peak-gflops "$PEAK_GFLOPS" --bandwidth-gbs "$BANDWIDTH_GBS"
echo "Timing, profiles, and roofline plots saved in $output_folder"
