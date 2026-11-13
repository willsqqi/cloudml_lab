# Project 1: Performance Modeling and Analysis

- `Project1-Report-Outline.docx`: project report.
- `resultscalculations.xlsx`: report tables, measured inputs, model calculations, and formulas linking the results to their sources.
- `code/`: scripts for running the experiment, collecting measurements, and drawing the Roofline plots.
- `A100/` and `L4/`: results for each GPU. Both contain `ResNet-50/`, `ResNet-101/`, and `ResNet-152/` subfolders.

Each model subfolder contains:

- `resnet*.json`: measured training time, throughput, peak memory use, and run settings.
- `resnet*-forward.csv` and `resnet*-backward.csv`: profiler measurements of calculations, memory transfers, and kernel durations for each pass.
- `roofline-points.csv`: calculated Roofline values for individual kernels, derived from the profiler CSVs.

Each GPU folder also contains `roofline-summary.csv`, which combines the kernel measurements into one result per model and pass, and `roofline.png`, which plots those calculated results.
