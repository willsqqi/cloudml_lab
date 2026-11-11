#!/usr/bin/env bash
set -euo pipefail

project_home="${1:-/home/sq2326}"
project_folder="$project_home/Project1-Performance-Modeling"
job_record="$project_home/project1-batch-job-28481.txt"
job_name="resnet-28481"
trap 'echo "Setup failed at line $LINENO. No successful run is implied; check this log and the job record." >&2' ERR

echo "Starting Project 1 setup at $(date -Is) on $(hostname)"
# prevent duplicate submission if the setup session restarts
exec 9> "$project_home/project1-autosubmit-28481.lock"
flock -n 9 || exit 0
if [[ -s "$job_record" ]]; then
    echo "Batch job already recorded: $(cat "$job_record")"
    exit 0
fi
existing_job=$(squeue -h -u "$(id -un)" --name="$job_name" -o '%A')
if [[ -n "$existing_job" ]]; then
    printf '%s\n' "$existing_job" > "$job_record"
    echo "Batch job already queued: $existing_job"
    exit 0
fi

# verify the uploaded bundle before extracting code and images
archive="$project_home/project1-batch-bundle.tar.gz"
printf '%s  %s\n' '17d70d0675544914b13ac0908bf2908b2e6e65bbf6bddf11924f05ceb8ee83c1' "$archive" | sha256sum -c -
tar -xzf "$archive" -C "$project_home"
cd "$project_folder/code"

# install a matching CUDA-enabled pair in a private environment
python_executable="$project_home/project1-py39-cu118/bin/python"
if [[ ! -x "$python_executable" ]]; then
    /share/apps/pyenv/py3.9/bin/python -m venv "$project_home/project1-py39-cu118"
fi
if ! "$python_executable" -c 'import torch, torchvision; assert torch.__version__ == "2.6.0+cu118" and torchvision.__version__ == "0.21.0+cu118"'; then
    "$python_executable" -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
fi

# check the imports and actual image loading before requesting a GPU
"$python_executable" - <<'PY'
from pathlib import Path
import torch
import torchvision
from main import load_batches
assert torch.version.cuda == "11.8", "A CUDA-enabled PyTorch build is required"
batches, images = load_batches(Path('../data/imagenet128'))
assert len(images) == 128 and len(batches) == 4
assert all(tuple(x.shape) == (32, 3, 224, 224) for x, y in batches)
print('Preflight passed:', torch.__version__, torchvision.__version__, len(images), 'images')
PY
bash -n run.slurm
job_id=$(sbatch --parsable --job-name="$job_name" --account=csci_ga_3033_085-2026fa --partition=g2-standard-12 run.slurm "$python_executable" "$project_folder/data/imagenet128")
printf '%s\n' "$job_id" > "$job_record"
echo "Submitted GPU batch job $job_id"
echo "Job record: $job_record"
echo "Logs: $project_folder/code/slurm-${job_id%%;*}.out and .err"
