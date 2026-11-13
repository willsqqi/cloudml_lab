#!/usr/bin/env bash
set -euo pipefail

# GCP startup hook: bash /path/to/code/profile.sh /path/to/images /path/to/L4
# Stage the code and images on the VM first. Use Linux with Python 3.9-3.12,
# python3-venv, an NVIDIA driver, and Nsight Compute 2024.4+ installed.
# Export PEAK_GFLOPS, BANDWIDTH_GBS, and NCU (if ncu is not on PATH) in the hook.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
data_folder="${1:?Usage: bash profile.sh DATA_FOLDER OUTPUT_FOLDER}"
output_folder="${2:?Choose a new GPU output folder; existing results are never overwritten}"

# A reboot must not silently repeat a paid experiment, even after a failure.
# To retry intentionally, archive .startup-run and the old logs first.
exec 9> .startup.lock
flock -n 9 || exit 0
if [[ -e .startup-run ]]; then
    echo "Startup already attempted. Results location: $(cat .startup-run)"
    exit 0
fi
printf '%s\n' "$output_folder" > .startup-run
echo "Setup log: $PWD/setup.log"
exec >> setup.log 2>&1
trap 'status=$?; echo "Setup/pipeline finished with exit code $status at $(date -Is)"; exit "$status"' EXIT
echo "Starting setup on $(hostname) at $(date -Is)"

: "${PEAK_GFLOPS:?Set PEAK_GFLOPS for the selected GPU}"
: "${BANDWIDTH_GBS:?Set BANDWIDTH_GBS for the selected GPU}"
export PEAK_GFLOPS BANDWIDTH_GBS
command -v "${NCU:-ncu}" >/dev/null || {
    echo "Nsight Compute is missing. Install ncu 2024.4+ or set NCU to its path." >&2
    exit 1
}
command -v timeout >/dev/null
nvidia-smi
[[ -f main.py && -f roofline.py && -f run.sh && -d "$data_folder" ]] || {
    echo "Missing code or images. Stage the code and the supplied image folder on the VM before startup." >&2
    exit 1
}

# Keep the same Python packages on both GPUs in a private environment.
python_executable="$PWD/.venv/bin/python"
if [[ ! -x "$python_executable" ]]; then
    python3 -c 'import sys; assert (3, 9) <= sys.version_info[:2] < (3, 13), "Use Python 3.9-3.12"'
    python3 -m venv .venv
fi
"$python_executable" -m pip install --timeout 60 --retries 2 \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
"$python_executable" -m pip install --timeout 60 --retries 2 \
    matplotlib==3.9.4 --index-url https://pypi.org/simple

# Limit the experiment process. This does NOT stop VM billing: configure GCP's
# maximum VM run time with termination action STOP before launching the VM.
# Persistent disks can still incur charges after a VM stops.
timeout --kill-after=30s "${MAX_RUN_TIME:-3h}" \
    bash run.sh "$python_executable" "$data_folder" "$output_folder"
