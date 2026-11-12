"""Plot model-level forward/backward summaries from Nsight Compute CSV exports."""

import argparse
import csv
import math
from pathlib import Path

FP32_ADD = "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
FP32_MUL = "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
FP32_FMA = "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
DRAM_READ = "dram__bytes_read.sum"
DRAM_WRITE = "dram__bytes_write.sum"
KERNEL_TIME = "gpu__time_duration.sum"
REQUIRED_METRICS = [FP32_ADD, FP32_MUL, FP32_FMA, DRAM_READ, DRAM_WRITE, KERNEL_TIME]
MODELS = ["resnet50", "resnet101", "resnet152"]
PHASES = ["forward", "backward"]


def value_in_base_units(value, unit):
    unit = unit.strip()
    unit_scales = {"": 1}
    for base in ("byte", "inst", "cycle", "second"):
        for prefix, scale in (("", 1), ("K", 1e3), ("M", 1e6), ("G", 1e9),
                              ("T", 1e12), ("m", 1e-3), ("u", 1e-6), ("n", 1e-9)):
            unit_scales[prefix + base] = scale
    if unit not in unit_scales:
        raise ValueError(f"Unsupported Nsight unit: {unit!r}; export with --print-units base")
    converted_value = float(value.replace(",", "")) * unit_scales[unit]
    if not math.isfinite(converted_value) or converted_value < 0:
        raise ValueError(f"Invalid metric value: {value} {unit}")
    return converted_value


def read_kernels(path):
    """Accept long or wide CSV layout; keep measurements together by launch ID."""
    text = path.read_text()
    if "==ERROR==" in text:
        raise ValueError(f"Nsight Compute reported an error in {path}")
    rows = iter(csv.reader(text.splitlines()))
    header = next((row for row in rows if "ID" in row and "Kernel Name" in row), None)
    if header is None:
        raise ValueError(f"No Nsight Compute CSV header in {path}")
    # Launch IDs distinguish repeated calls to the same kernel.
    kernels_by_id = {}
    long_format = "Metric Name" in header
    units = {} if long_format else dict(zip(header, next(rows)))
    for row in rows:
        record = dict(zip(header, row))
        if not record.get("ID", "").isdigit():
            continue
        kernel_metrics = kernels_by_id.setdefault(record["ID"], {"kernel": record["Kernel Name"]})
        if long_format:
            metric = record["Metric Name"]
            if metric in REQUIRED_METRICS:
                kernel_metrics[metric] = value_in_base_units(record["Metric Value"], record["Metric Unit"])
        else:
            for metric in REQUIRED_METRICS:
                kernel_metrics[metric] = value_in_base_units(record[metric], units[metric])
    if not kernels_by_id:
        raise ValueError(f"No profiled kernels in {path}")
    for kernel_metrics in kernels_by_id.values():
        if any(metric not in kernel_metrics for metric in REQUIRED_METRICS) or kernel_metrics[KERNEL_TIME] <= 0:
            raise ValueError(f"Missing counters or duration in {path}: {kernel_metrics['kernel']}")
    return kernels_by_id.values()


def main():
    """Read profiling files, calculate the points, and save the plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="One GPU's six model-phase CSV files")
    parser.add_argument("--peak-gflops", type=float, required=True, help="This GPU's FP32 peak, GFLOP/s")
    parser.add_argument("--bandwidth-gbs", type=float, required=True, help="This GPU's DRAM bandwidth, GB/s")
    args = parser.parse_args()
    if any(not math.isfinite(limit) or limit <= 0 for limit in [args.peak_gflops, args.bandwidth_gbs]):
        parser.error("Peak compute and memory bandwidth must be positive finite numbers.")

    roofline_points = []
    for model in MODELS:
        for phase in PHASES:
            for kernel_metrics in read_kernels(args.folder / f"{model}-{phase}.csv"):
                # Each fused multiply-add (FMA) performs two floating-point operations.
                flops = kernel_metrics[FP32_ADD] + kernel_metrics[FP32_MUL] + 2 * kernel_metrics[FP32_FMA]
                dram_bytes = kernel_metrics[DRAM_READ] + kernel_metrics[DRAM_WRITE]
                roofline_points.append({
                    "model": model, "phase": phase, "kernel": kernel_metrics["kernel"],
                    "flops": flops, "dram_bytes": dram_bytes, "seconds": kernel_metrics[KERNEL_TIME],
                    "flops_per_byte": flops / dram_bytes if dram_bytes else 0,
                    "gflops_per_second": flops / kernel_metrics[KERNEL_TIME] / 1e9,
                })
    model_summaries = []
    for model in MODELS:
        for phase in PHASES:
            kernels = [point for point in roofline_points
                       if point["model"] == model and point["phase"] == phase]
            # Sum all kernels before dividing, including zero-FLOP or zero-traffic kernels.
            total_flops = sum(point["flops"] for point in kernels)
            total_bytes = sum(point["dram_bytes"] for point in kernels)
            kernel_seconds = sum(point["seconds"] for point in kernels)
            model_summaries.append({
                "model": model, "phase": phase, "kernel_count": len(kernels),
                "flops": total_flops, "dram_bytes": total_bytes,
                "summed_kernel_seconds": kernel_seconds,
                "flops_per_byte": total_flops / total_bytes if total_bytes else 0,
                "gflops_per_second": total_flops / kernel_seconds / 1e9,
            })
    # Summed profiled kernel durations are not end-to-end phase or training-step time.
    plotted_points = [point for point in model_summaries
                      if point["flops_per_byte"] > 0 and point["gflops_per_second"] > 0]
    if not plotted_points:
        raise ValueError("No positive FLOP/byte measurements to plot.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    intensities = [point["flops_per_byte"] for point in plotted_points]
    ridge_intensity = args.peak_gflops / args.bandwidth_gbs
    min_exponent = math.floor(math.log10(min(*intensities, ridge_intensity))) - 1
    max_exponent = math.ceil(math.log10(max(*intensities, ridge_intensity))) + 1
    intensity_values = [10 ** (min_exponent + (max_exponent - min_exponent) * i / 200)
                        for i in range(201)]
    # The roof is the lower of the compute limit and the memory-bandwidth limit.
    performance_bound = [min(args.peak_gflops, args.bandwidth_gbs * intensity)
                         for intensity in intensity_values]
    figure, panels = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for panel, phase in zip(panels, PHASES):
        panel.loglog(intensity_values, performance_bound, color="black", label="FP32 / DRAM roof")
        for model, color in zip(MODELS, ["tab:blue", "tab:orange", "tab:green"]):
            model_points = [point for point in plotted_points
                            if point["model"] == model and point["phase"] == phase]
            panel.scatter([point["flops_per_byte"] for point in model_points],
                          [point["gflops_per_second"] for point in model_points],
                          s=65, color=color, label=model)
        panel.set_title(phase.capitalize())
        panel.set_xlabel("Arithmetic intensity (FLOP/byte)")
        panel.grid(True, which="both", alpha=0.2)
    panels[0].set_ylabel("Aggregate profiled kernel performance (GFLOP/s)")
    panels[1].legend(fontsize=8)
    figure.suptitle(f"{args.folder.name}: one point per model in each phase")
    figure.tight_layout()
    figure.savefig(args.folder / "roofline.png", dpi=200)
    plt.close(figure)
    for filename, rows in [("roofline-points.csv", roofline_points),
                           ("roofline-summary.csv", model_summaries)]:
        with (args.folder / filename).open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved roofline.png, roofline-points.csv, and roofline-summary.csv in {args.folder}")
    print(f"Plotted {len(plotted_points)} of {len(model_summaries)} model/phase summaries; omitted zero-work/zero-traffic summaries.")


if __name__ == "__main__":
    main()
