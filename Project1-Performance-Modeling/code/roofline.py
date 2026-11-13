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
    for base in ("byte", "inst", "cycle", "second", "s"):
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
    parser.add_argument("folder", type=Path, help="GPU folder containing ResNet-50, ResNet-101, and ResNet-152 subfolders")
    parser.add_argument("--peak-gflops", type=float, required=True, help="This GPU's FP32 peak, GFLOP/s")
    parser.add_argument("--bandwidth-gbs", type=float, required=True, help="This GPU's DRAM bandwidth, GB/s")
    parser.add_argument("--title", help="Figure title, such as 'NVIDIA L4' (default: the folder name)")
    args = parser.parse_args()
    if any(not math.isfinite(limit) or limit <= 0 for limit in [args.peak_gflops, args.bandwidth_gbs]):
        parser.error("Peak compute and memory bandwidth must be positive finite numbers.")

    roofline_points = []
    for model in MODELS:
        model_folder = args.folder / f"ResNet-{model[6:]}"
        for phase in PHASES:
            for kernel_metrics in read_kernels(model_folder / f"{model}-{phase}.csv"):
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
    from matplotlib.ticker import FuncFormatter

    # Use the same axes for every GPU so the figures can be compared side by side.
    ridge_intensity = args.peak_gflops / args.bandwidth_gbs
    intensities = [point["flops_per_byte"] for point in plotted_points] + [ridge_intensity]
    rates = [point["gflops_per_second"] for point in plotted_points] + [args.peak_gflops]
    x_limits = (min(1.0, min(intensities) / 2), max(1000.0, max(intensities) * 2))
    y_limits = (min(200.0, min(rates) / 2), max(50000.0, max(rates) * 1.5))
    steps = 400
    intensity_values = [x_limits[0] * (x_limits[1] / x_limits[0]) ** (i / steps) for i in range(steps + 1)]
    # The roof is the lower of the compute limit and the memory-bandwidth limit.
    performance_bound = [min(args.peak_gflops, args.bandwidth_gbs * intensity)
                         for intensity in intensity_values]
    model_styles = {"resnet50": ("#0072B2", "o", "ResNet-50"),
                    "resnet101": ("#D55E00", "s", "ResNet-101"),
                    "resnet152": ("#009E73", "^", "ResNet-152")}
    number_format = FuncFormatter(lambda value, _: f"{value:,.0f}")

    figure, panels = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True, sharey=True)
    for panel, phase in zip(panels, PHASES):
        panel.plot(intensity_values, performance_bound, color="#333333", lw=2, zorder=2)
        panel.axvline(ridge_intensity, color="#777777", ls=":", lw=1, zorder=1)
        for model in MODELS:
            color, marker, label = model_styles[model]
            model_points = [point for point in plotted_points
                            if point["model"] == model and point["phase"] == phase]
            panel.scatter([point["flops_per_byte"] for point in model_points],
                          [point["gflops_per_second"] for point in model_points],
                          s=90, marker=marker, color=color, edgecolors="white", linewidths=0.8,
                          label=label, zorder=3)
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlim(*x_limits)
        panel.set_ylim(*y_limits)
        panel.set_yticks([tick for tick in (200, 1000, 10000, 50000) if y_limits[0] <= tick <= y_limits[1]])
        panel.xaxis.set_major_formatter(number_format)
        panel.yaxis.set_major_formatter(number_format)
        panel.set_title(phase.capitalize(), fontsize=13)
        panel.set_xlabel("Arithmetic intensity (FLOP/byte)", fontsize=12)
        panel.tick_params(labelsize=11)
        panel.grid(True, which="major", color="#e6e6e6", lw=0.8)
        panel.set_axisbelow(True)
        panel.spines[["top", "right"]].set_visible(False)
    panels[0].set_ylabel("Aggregate kernel rate (GFLOP/s)", fontsize=12)
    handles, labels = panels[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=12,
                  bbox_to_anchor=(0.5, 0.0))
    figure.suptitle(args.title or args.folder.name, fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(args.folder / "roofline.png", dpi=200, facecolor="white")
    plt.close(figure)
    # Keep kernel details with each model and the comparison summary at GPU level.
    csv_outputs = [(args.folder / "roofline-summary.csv", model_summaries)]
    for model in MODELS:
        model_folder = args.folder / f"ResNet-{model[6:]}"
        csv_outputs.append((model_folder / "roofline-points.csv",
                            [point for point in roofline_points if point["model"] == model]))
    for path, rows in csv_outputs:
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved roofline.png and roofline-summary.csv in {args.folder}; kernel points are in each model folder")
    print(f"Plotted {len(plotted_points)} of {len(model_summaries)} model/phase summaries; omitted zero-work/zero-traffic summaries.")


if __name__ == "__main__":
    main()
