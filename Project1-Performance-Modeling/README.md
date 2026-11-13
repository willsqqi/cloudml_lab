# Project 1: Performance Modeling and Analysis

- `Project One Report.docx`: project report.
- `resultscalculations.xlsx`: report tables, measured inputs, model calculations, and formulas linking the results to their sources.
- `code/`: scripts for running the experiment, collecting measurements, and drawing the Roofline plots.
  `kernel_groups.py` groups the profiled kernels and produces Tables 5 and 6 and `kernel-roofline.png`.
- `kernel-roofline.png`: kernel-level Roofline for both GPUs (report Figure 5).
- `A100/` and `L4/`: results for each GPU. Both contain `ResNet-50/`, `ResNet-101/`, and `ResNet-152/` subfolders.

Each model subfolder contains:

- `resnet*.json`: measured training time, throughput, peak memory use, and run settings.
- `resnet*-forward.csv` and `resnet*-backward.csv`: profiler measurements of calculations, memory transfers, and kernel durations for each pass.
- `roofline-points.csv`: calculated Roofline values for individual kernels, derived from the profiler CSVs.

Each GPU folder also contains `roofline-summary.csv`, which combines the kernel measurements into one result per model and pass, and `roofline.png`, which plots those calculated results.

## Regenerating the report figures

All figures come from the saved results; no GPU is needed. Run from this folder:

```
python code/roofline.py L4 --peak-gflops 30300 --bandwidth-gbs 300 --title "NVIDIA L4"                   # Figure 3
python code/roofline.py A100 --peak-gflops 19500 --bandwidth-gbs 1555 --title "NVIDIA A100 SXM4 40 GB"   # Figure 4
python code/kernel_groups.py .                                                                             # Figure 5, Tables 5-6
```
