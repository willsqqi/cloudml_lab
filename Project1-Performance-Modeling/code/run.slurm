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

if [[ ! -f main.py || ! -d "$data_folder" ]]; then
    echo "Run from the code folder and make sure $data_folder is available." >&2
    exit 1
fi

# keep results and environment details together for each job
mkdir -p "$output_folder"
nvidia-smi > "$output_folder/gpu.txt"
"$python_executable" -m pip freeze > "$output_folder/packages.txt"

# run the three models one at a time on the allocated gpu
for model in resnet50 resnet101 resnet152; do
    echo "Running $model"
    "$python_executable" -u main.py --data "$data_folder" --model "$model" --output "$output_folder"
done

echo "Results saved in $output_folder"
