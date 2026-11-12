# Project 1 experiment code

Four scripts:

- `main.py`: measure training time, throughput and peak allocated GPU memory; mark forward and backward for profiling.
- `roofline.py`: read Nsight Compute CSV files and draw forward/backward panels with one point per model for one GPU.
- `run.slurm`: queue timing, six Nsight profiles, and roofline plotting automatically.
- `profile.sh`: prepare the environment and submit one L4 batch job from the OOD startup hook.

`main()` is the entry point in each Python script. In `main.py`, two small helpers prepare the images and run one training step.

Submit the batch script from a shell connected to the Burst cluster. Slurm runs it on the allocated GPU compute node. Use the same CUDA-enabled PyTorch, Torchvision and Matplotlib environment on L4 and A100. Profiling uses NVIDIA Nsight Compute (`ncu`, version 2024.4 or newer) and requires access to GPU performance counters.

## Settings

The constants at the top of `main.py` set batch size 32, a fixed subset of 128 images, 10 warm-up steps, 50 measured steps and 3 repetitions. Images use a 224 × 224 center crop. The code uses FP32 with TF32 disabled, random model initialization, and SGD with learning rate 0.01 and momentum 0.9.

Each repetition restores the initial weights and resets the optimizer before warm-up. Timing includes gradient reset, forward, loss, backward and optimizer update. The four image batches are loaded onto the GPU beforehand and reused, so timing excludes data loading and transfer. This measures GPU-resident training performance, not accuracy or time to convergence. Peak allocated memory includes those resident batches.

Use the prepared 128-image [Imagenette subset](../data/README.md) on both GPUs, with one subfolder per class. The script loads these images in a fixed order using seed 0 and saves their paths and labels in its JSON output. Compare these lists across runs to confirm the same input subset. The experiment script does not download data or pretrained weights.

## Submit a batch job

Place `code/`, the prepared `data/imagenet128/` folder, and a CUDA-enabled Python environment on storage visible to the Burst compute nodes. Torch and Burst have [separate compute storage](https://services.rt.nyu.edu/docs/cloud/hpc_bursting_to_cloud/intro/). Slurm transfers the batch script itself; it does not upload the Python files, images or environment.

From the `code` folder in an OOD Jupyter terminal, activate your Python environment. Before submission, export `PEAK_GFLOPS` and `BANDWIDTH_GBS` as positive numbers for the exact GPU board (ordinary FP32, not Tensor Core throughput). These settings are inherited by the batch job. If `ncu` is not on PATH, export `NCU` as its absolute path. Then submit for the matching GPU type:

```bash
sbatch run.slurm                                  # L4
sbatch --partition=c12m85-a100-1 run.slurm          # A100 40 GB
```

The script requests one GPU, four CPU cores, 16 GB of host memory and a one-hour time limit using the course account. Each job runs ResNet-50, ResNet-101 and ResNet-152 in sequence, then exits. It stops if a command fails. Timing runs are followed by six separate Nsight Compute runs and automatic plotting. Kernel replay can take substantially longer than timing alone; the one-hour limit must be checked on HPC and increased at submission if needed.

To select Python and the image folder explicitly, pass them as arguments:

```bash
sbatch run.slurm /absolute/path/to/environment/bin/python /absolute/path/to/imagenet128
```

Use a Python environment containing CUDA-enabled PyTorch, Torchvision, Matplotlib and pip. The cluster environment still needs verification; the script does not install packages. Use the same environment for both GPUs.

After submission, you can disconnect. Slurm starts the script when its resources are ready. Results go into `results/<partition>-<job-id>/`, including GPU details, package and NCU versions, three timing JSON files, six `{model}-{phase}.csv` files, six `.ncu-rep` reports, `roofline.png`, `roofline-points.csv`, and `roofline-summary.csv`. Profile metadata is saved separately under `profile-metadata/<phase>/`. A missing profiler, unavailable counters, or invalid CSV stops the job instead of reporting success. Console output and errors go into `slurm-<job-id>.out` and `slurm-<job-id>.err` in the submission folder.

```bash
squeue --me                                      # queued and running jobs
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
```

Replace `JOB_ID` with the number returned by `sbatch`. See the [Slurm batch submission documentation](https://slurm.schedmd.com/sbatch.html) for submission and output behavior.

## Automatic startup from OOD

Upload `profile.sh` to `/home/sq2326/profile.sh` and the matching `project1-pipeline-bundle.tar.gz` to `/home/sq2326/`. The bundle contains `main.py`, `roofline.py`, `run.slurm`, and the images; it deliberately excludes the host bootstrapper. Its SHA-256 must match the checksum in `profile.sh`. Rebuild the bundle and update the checksum whenever its code or data changes; old uploaded bundles still contain the old timing-only pipeline.

In the session's startup hook, export `PEAK_GFLOPS` and `BANDWIDTH_GBS` with the verified board specifications, then invoke:

```bash
bash /home/sq2326/profile.sh > /home/sq2326/project1-autosubmit-pipeline.log 2>&1 &
```

`PARTITION` defaults to `g2-standard-12`; set it to `c12m85-a100-1` for A100 and use matching peak/bandwidth values. The hook must also export `NCU` if a full profiler path is needed. The bootstrapper installs the Python dependencies, checks image loading, and submits the complete batch pipeline. It does not install Nsight Compute or grant counter permissions.

The lock and `project1-batch-job-pipeline.txt` prevent duplicate submission. An existing job record deliberately stops further submissions, even after failure; inspect the recorded job before explicitly archiving the record for a retry or the other GPU. Do not attach multiple startup hooks for the same run.

## Optional manual procedures

The following commands are for troubleshooting individual stages; `run.slurm` now performs timing, profiling, and plotting automatically.

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

Replace the two placeholders with numbers, then repeat for `results/a100` using that board's values. The script expects all six model-phase CSV files in each folder. It produces `roofline.png` with forward and backward panels, each containing one point for each of the three ResNet models. Model colors are consistent across phases and GPUs. Both panels within each figure share the same roof and axis scales; two GPU runs produce four panels in total.

`roofline-summary.csv` contains six model/phase rows, including kernel counts, total FLOPs, total DRAM bytes, summed kernel durations, and the calculated coordinates. `roofline-points.csv` retains the per-kernel measurements. The six input CSVs and original `.ncu-rep` files are unchanged and can be used for detailed bottleneck analysis.

FLOPs are FP32 additions + multiplications + twice fused multiply-adds. Bytes are DRAM reads + writes. For each model and phase, sum these quantities over all captured kernels before calculating arithmetic intensity (total FLOPs / total bytes) and performance (total FLOPs / summed kernel seconds). Do not average the individual kernel coordinates. Include zero-FLOP and zero-traffic kernels in the totals; summaries with zero total FLOPs or bytes remain in the CSV but are omitted from the logarithmic plot, with a count printed. These instruction counts exclude special-function and integer operations.

The figure reports aggregate profiled kernel performance. Each point uses FLOPs, bytes, and durations from the same model/phase capture. Summed kernel durations are not elapsed phase time: they exclude gaps and do not account for overlap between kernels. Check a timeline and account for overlap before interpreting elapsed phase performance. Profiling replay also affects execution, so use the separate unprofiled training-step JSON measurements for training throughput.

API references: [PyTorch CUDA timing and FP32 settings](https://docs.pytorch.org/docs/stable/notes/cuda), [Nsight Compute CLI and NVTX filtering](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html), and [NVIDIA FLOP metric mapping](https://docs.nvidia.com/cupti/13.5.0/main/main.html).
