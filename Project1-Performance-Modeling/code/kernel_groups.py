"""Reproduce the kernel-group tables (report Tables 5 and 6) and the kernel-level Roofline figure
from the saved roofline points. No GPU is needed.

Usage: python kernel_groups.py PROJECT_FOLDER   (the folder that contains A100/ and L4/)
Writes PROJECT_FOLDER/kernel-roofline.png and prints the table values.
"""
import csv
import json
import math
import sys
from pathlib import Path

SPECS = {"L4": (30300, 300), "A100": (19500, 1555)}  # peak FP32 GFLOP/s, DRAM GB/s
MODELS = ["50", "101", "152"]


def kernel_group(name):
    """Convolution/GEMM kernels versus batch-norm and element-wise kernels."""
    n = name.lower()
    if "pool" in n or "reduce_kernel" in n or "nll_loss" in n:
        return "other"
    if "bn_" in n or "elementwise" in n or n.startswith("void at::"):
        return "memory"
    return "conv"


def summarize(folder, gpu, model):
    model_folder = folder / gpu / f"ResNet-{model}"
    totals = {g: {"flops": 0.0, "bytes": 0.0, "seconds": 0.0} for g in ("conv", "memory", "other")}
    best_rate = 0.0
    for row in csv.DictReader(open(model_folder / "roofline-points.csv")):
        group = totals[kernel_group(row["kernel"])]
        flops, nbytes, seconds = float(row["flops"]), float(row["dram_bytes"]), float(row["seconds"])
        group["flops"] += flops
        group["bytes"] += nbytes
        group["seconds"] += seconds
        if seconds > 100e-6:  # ignore very short kernels when finding the fastest one
            best_rate = max(best_rate, flops / seconds)
    timing = json.load(open(model_folder / f"resnet{model}.json"))
    steps = [m["step_ms"] for m in timing["measurements"]]
    return totals, best_rate, sum(steps) / len(steps)


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    results = {(g, m): summarize(folder, g, m) for g in SPECS for m in MODELS}
    print("GPU   Model  conv%   conv GFLOP/s (%peak)   mem%   mem GB/s (%peak)   kernel/step   best kernel")
    for gpu, (peak, bandwidth) in SPECS.items():
        for model in MODELS:
            totals, best, step_ms = results[(gpu, model)]
            total_s = sum(t["seconds"] for t in totals.values())
            conv, mem = totals["conv"], totals["memory"]
            conv_rate = conv["flops"] / conv["seconds"] / 1e9
            mem_bw = mem["bytes"] / mem["seconds"] / 1e9
            print(f"{gpu:5} {model:>5}  {conv['seconds'] / total_s:6.1%}  {conv_rate:8,.0f} ({conv_rate / peak:4.0%})"
                  f"      {mem['seconds'] / total_s:6.1%}  {mem_bw:6,.0f} ({mem_bw / bandwidth:4.0%})"
                  f"     {total_s * 1e3 / step_ms:6.1%}     {best / 1e9:,.0f} ({best / 1e9 / peak:.0%})")
    print("\nModel  step speedup  kernel speedup  conv speedup  mem speedup  mem share of L4 extra time")
    for model in MODELS:
        a_tot, _, a_step = results[("A100", model)]
        l_tot, _, l_step = results[("L4", model)]
        a_s = sum(t["seconds"] for t in a_tot.values())
        l_s = sum(t["seconds"] for t in l_tot.values())
        extra = l_s - a_s
        print(f"{model:>5}  {l_step / a_step:11.2f}x  {l_s / a_s:13.2f}x  "
              f"{l_tot['conv']['seconds'] / a_tot['conv']['seconds']:11.2f}x  "
              f"{l_tot['memory']['seconds'] / a_tot['memory']['seconds']:10.2f}x  "
              f"{(l_tot['memory']['seconds'] - a_tot['memory']['seconds']) / extra:12.0%}")


GROUP_STYLE = {  # validated categorical slots 1 and 2
    "conv": ("#2a78d6", "Convolution / GEMM kernels"),
    "memory": ("#eb6834", "Batch-norm and element-wise kernels"),
}
GPU_TITLE = {"L4": "NVIDIA L4", "A100": "NVIDIA A100 SXM4 40 GB"}


def read_points(folder, gpu):
    """Every profiled kernel of all three models, forward and backward."""
    points = []
    for model in MODELS:
        for row in csv.DictReader(open(folder / gpu / f"ResNet-{model}" / "roofline-points.csv")):
            points.append({"model": model, "phase": row["phase"], "group": kernel_group(row["kernel"]),
                           "flops": float(row["flops"]), "bytes": float(row["dram_bytes"]),
                           "seconds": float(row["seconds"])})
    return points


def plot(folder, out_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    figure, panels = plt.subplots(1, 2, figsize=(11, 4.9), sharex=True, sharey=True)
    x_range, y_range = (1e-2, 1e3), (1e1, 5e4)
    for panel, gpu in zip(panels, ["L4", "A100"]):
        peak, bandwidth = SPECS[gpu]
        points = read_points(folder, gpu)
        plotted = [p for p in points if p["flops"] > 0 and p["group"] in GROUP_STYLE]
        zero_flop_share = (sum(p["seconds"] for p in points if p["flops"] == 0)
                           / sum(p["seconds"] for p in points))
        longest = max(p["seconds"] for p in plotted)
        # kernels: marker area proportional to kernel duration
        for group, (color, _) in GROUP_STYLE.items():
            sel = [p for p in plotted if p["group"] == group]
            panel.scatter([p["flops"] / p["bytes"] for p in sel],
                          [p["flops"] / p["seconds"] / 1e9 for p in sel],
                          s=[6 + 110 * p["seconds"] / longest for p in sel],
                          color=color, alpha=0.45, linewidths=0.4, edgecolors="#fcfcfb", zorder=2)
        # published roof
        xs = [10 ** (math.log10(x_range[0]) + i * 5 / 400) for i in range(401)]
        panel.plot(xs, [min(peak, bandwidth * x) for x in xs], color="#0b0b0b", lw=1.6, zorder=3)
        # best measured kernel rate (kernels longer than 100 us)
        best = max(p["flops"] / p["seconds"] for p in plotted if p["seconds"] > 100e-6) / 1e9
        panel.plot(x_range, [best, best], color="#52514e", lw=1.2, ls=":", zorder=3)
        panel.text(x_range[0] * 1.3, best / 1.15, f"fastest measured kernel\n{best / 1e3:.1f} TFLOP/s, {best / peak:.0%} of peak", ha="left", va="top", fontsize=8, color="#52514e")
        panel.text(x_range[1] / 1.15, peak * 1.12, f"published FP32 peak {peak / 1e3:.1f} TFLOP/s",
                   ha="right", va="bottom", fontsize=8, color="#0b0b0b")
        # model-level points from Table 4 (one per model and phase)
        for model in MODELS:
            for phase in ("forward", "backward"):
                sel = [p for p in points if p["model"] == model and p["phase"] == phase]
                f = sum(p["flops"] for p in sel)
                panel.scatter(f / sum(p["bytes"] for p in sel), f / sum(p["seconds"] for p in sel) / 1e9,
                              marker="X", s=46, color="#0b0b0b", edgecolors="#fcfcfb", linewidths=0.8,
                              zorder=5)
        # group totals across all three models
        for group, (color, _) in GROUP_STYLE.items():
            sel = [p for p in points if p["group"] == group]
            f = sum(p["flops"] for p in sel)
            panel.scatter(f / sum(p["bytes"] for p in sel), f / sum(p["seconds"] for p in sel) / 1e9,
                          marker="D", s=90, color=color, edgecolors="#0b0b0b", linewidths=1.2, zorder=6)
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlim(*x_range)
        panel.set_ylim(*y_range)
        panel.grid(True, which="major", color="#e4e3df", lw=0.6)
        panel.set_axisbelow(True)
        panel.set_title(GPU_TITLE[gpu], fontsize=11, fontweight="bold", color="#0b0b0b")
        panel.set_xlabel("Arithmetic intensity (FLOP/byte)")
        panel.text(x_range[1] / 1.3, y_range[0] * 1.3,
                   f"Not shown: kernels with no FP32 arithmetic\n({zero_flop_share:.0%} of kernel time)",
                   ha="right", va="bottom", fontsize=8, color="#52514e")
    panels[0].set_ylabel("Kernel performance (GFLOP/s)")
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=c, alpha=0.6, label=l)
               for c, l in GROUP_STYLE.values()]
    handles += [Line2D([], [], ls="", marker="D", ms=7, color="#9a9994", mec="#0b0b0b", mew=1.2,
                       label="Group total (all three models)"),
                Line2D([], [], ls="", marker="X", ms=7, color="#0b0b0b", label="Model-level point (Table 4)"),
                Line2D([], [], color="#0b0b0b", lw=1.6, label="Published roof"),
                Line2D([], [], color="#52514e", lw=1.2, ls=":", label="Fastest measured kernel")]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
                  bbox_to_anchor=(0.5, -0.01))
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    output = (out_dir or folder) / "kernel-roofline.png"
    figure.savefig(output, dpi=200, facecolor="#ffffff")
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
    plot(Path(sys.argv[1] if len(sys.argv) > 1 else "."),
         Path(sys.argv[2]) if len(sys.argv) > 2 else None)
