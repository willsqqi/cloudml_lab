# Project 1 experiment code

Four scripts:

- `main.py`: measure training time, throughput and peak allocated GPU memory; mark forward and backward for profiling.
- `roofline.py`: read Nsight Compute CSV files and draw separate forward/backward panels for one GPU.
- `run.slurm`: queue a GPU job and run all three models automatically.
- `profile.sh`: prepare the environment and submit one L4 batch job from the OOD startup hook.

`main()` is the entry point in each Python script. In `main.py`, two small helpers prepare the images and run one training step.

Submit the batch script from a shell connected to the Burst cluster. Slurm runs it on the allocated GPU compute node. Use the same CUDA-enabled PyTorch, Torchvision and Matplotlib environment on L4 and A100. Profiling uses NVIDIA Nsight Compute (`ncu`, version 2024.4 or newer) and requires access to GPU performance counters.

## Settings

The constants at the top of `main.py` set batch size 32, a fixed subset of 128 images, 10 warm-up steps, 50 measured steps and 3 repetitions. Images use a 224 × 224 center crop. The code uses FP32 with TF32 disabled, random model initialization, and SGD with learning rate 0.01 and momentum 0.9.

Each repetition restores the initial weights and resets the optimizer before warm-up. Timing includes gradient reset, forward, loss, backward and optimizer update. The four image batches are loaded onto the GPU beforehand and reused, so timing excludes data loading and transfer. This measures GPU-resident training performance, not accuracy or time to convergence. Peak allocated memory includes those resident batches.

Use the prepared 128-image [Imagenette subset](../data/README.md) on both GPUs, with one subfolder per class. The script loads these images in a fixed order using seed 0 and saves their paths and labels in its JSON output. Compare these lists across runs to confirm the same input subset. The experiment script does not download data or pretrained weights.

## Submit a batch job

Place `code/`, the prepared `data/imagenet128/` folder, and a CUDA-enabled Python environment on storage visible to the Burst compute nodes. Torch and Burst have [separate compute storage](https://services.rt.nyu.edu/docs/cloud/hpc_bursting_to_cloud/intro/). Slurm transfers the batch script itself; it does not upload the Python files, images or environment.

From the `code` folder on Burst, activate your Python environment and submit one job for each GPU type:

```bash
sbatch run.slurm                                  # L4
sbatch --partition=c12m85-a100-1 run.slurm          # A100 40 GB
```

The script requests one GPU, four CPU cores, 16 GB of host memory and a one-hour time limit using the course account. Each job runs ResNet-50, ResNet-101 and ResNet-152 in sequence, then exits. It stops if a command fails. These are the unprofiled timing runs; the profiling procedure is below.

To select Python and the image folder explicitly, pass them as arguments:

```bash
sbatch run.slurm /absolute/path/to/environment/bin/python /absolute/path/to/imagenet128
```

Use a Python environment containing CUDA-enabled PyTorch, Torchvision and pip. The cluster environment still needs verification; the script does not install packages. Use the same environment for both GPUs.

After submission, you can disconnect. Slurm starts the script when its resources are ready. Results go into `results/<partition>-<job-id>/`, including the GPU details, package versions and three model JSON files. Console output and errors go into `slurm-<job-id>.out` and `slurm-<job-id>.err` in the submission folder.

```bash
squeue --me                                      # queued and running jobs
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
```

Replace `JOB_ID` with the number returned by `sbatch`. See the [Slurm batch submission documentation](https://slurm.schedmd.com/sbatch.html) for submission and output behavior.

## Automatic startup for OOD session 28481

`profile.sh` is the startup script installed at `/home/sq2326/profile.sh` for the current CPU-only OOD session. It checks and extracts the uploaded `project1-batch-bundle.tar.gz`, prepares a private environment with PyTorch 2.6.0 and Torchvision 0.21.0 for CUDA 11.8, checks image loading, then submits `run.slurm` to the L4 partition. A lock and job record prevent repeated submission.

The session's own `before.sh` calls it in the background:

```bash
bash /home/sq2326/profile.sh > /home/sq2326/project1-autosubmit-28481.log 2>&1 &
```

The setup log is `project1-autosubmit-28481.log`; the submitted GPU job ID is saved in `project1-batch-job-28481.txt`, both in `/home/sq2326`. Setup must finish successfully before the GPU job is submitted. This runs the timing benchmark; Nsight profiling remains a separate step below.

This copy is tied to session 28481 and the uploaded bundle's recorded checksum. Uploading or committing it does not attach it to future OOD sessions. Shell syntax and mock submission checks passed; execution on HPC remains unverified.

## Measure training manually

Set the dataset path and output folder for the allocated GPU:

```bash
DATA=../data/imagenet128
OUT=results/l4
mkdir -p "$OUT"
nvidia-smi > "$OUT/gpu.txt"
python -m pip freeze > "$OUT/packages.txt"

for model in resnet50 resnet101 resnet152; do
  python main.py --data "$DATA" --model "$model" --output "$OUT"
done
```

Repeat on A100 with `OUT=results/a100`. Each JSON contains the GPU and software versions, per-repetition measurements, and median/minimum/maximum values. GPU speedup is median L4 step time divided by median A100 step time for the same model.

## Profile forward and backward

Run profiling separately from the timing runs above. Only one step after warm-up is marked. The forward range contains the model call; backward contains `loss.backward()` (including loss gradients). The loss forward calculation and optimizer update are outside both ranges.

```bash
METRICS=smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum
ncu --version > "$OUT/ncu-version.txt"

for model in resnet50 resnet101 resnet152; do
  for phase in forward backward; do
    ncu --nvtx --nvtx-push-pop-scope process --nvtx-include "$phase/" \
      --metrics "$METRICS" --replay-mode kernel --clock-control none \
      --cache-control all --csv --page raw --print-units base \
      --print-metric-name name --log-file "$OUT/$model-$phase.csv" \
      --export "$OUT/$model-$phase" \
      python main.py --data "$DATA" --model "$model" --output "$OUT" --profile
  done
done
```

Keep the `.ncu-rep` files: they retain profiler settings and launch details. Cache flushing is enabled during profiling; these kernel durations are separate from unprofiled training-step timing. If counters are unavailable or profiling fails, resolve that before making plots; missing counters are not treated as zeros.

## Draw Roofline plots

Supply the FP32 compute peak in **GFLOP/s** and DRAM bandwidth in **GB/s** for the exact GPU board being used. Use ordinary FP32 peaks, since TF32 is disabled. For example, convert a peak in TFLOP/s to GFLOP/s by multiplying by 1,000.

```bash
python roofline.py results/l4 --peak-gflops <FP32_PEAK> --bandwidth-gbs <DRAM_BANDWIDTH>
```

Replace the two placeholders with numbers, then repeat for `results/a100` using that board's values. The script expects all six model-phase CSV files in each folder and produces `roofline.png` and `roofline-points.csv`. Both phase panels share the same roof and axis scales. Each point represents one kernel invocation; no model-level aggregation is applied.

FLOPs are FP32 additions + multiplications + twice fused multiply-adds. Bytes are DRAM reads + writes. Arithmetic intensity is FLOPs/bytes; performance is FLOPs/seconds. These instruction counts exclude special-function and integer operations. Zero-work or zero-traffic kernels are retained in the output CSV and omitted from the logarithmic plot, with a count printed.

API references: [PyTorch CUDA timing and FP32 settings](https://docs.pytorch.org/docs/stable/notes/cuda), [Nsight Compute CLI and NVTX filtering](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html), and [NVIDIA FLOP metric mapping](https://docs.nvidia.com/cupti/13.5.0/main/main.html).
