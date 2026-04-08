#!/usr/bin/env python3
"""
scaling_plot_3way.py — Three-way comparison: FJ Pure vs FJ+TP vs TaskInfer

Uses llama-batched-bench JSONL output for fair comparison across all systems.
"""

import json
import os
import glob
from statistics import median
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLOR_FJ = "#d62728"       # red — FJ Pure
COLOR_FJTP = "#ff7f0e"     # orange — FJ+TP
COLOR_TF = "#2ca02c"       # green — TaskInfer
COLOR_IDEAL = "#888888"    # gray


def load_results(data_dir):
    """Load JSONL results from all log files."""
    configs = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "*.log"))):
        fname = os.path.basename(f).replace(".log", "")
        parts = fname.split("_")
        numa = parts[1]
        system = parts[2]

        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("{"):
                    try:
                        d = json.loads(line)
                        key = (numa, system)
                        configs.setdefault(key, {"pp": [], "tg": [], "total": []})
                        configs[key]["pp"].append(d.get("speed_pp", 0))
                        configs[key]["tg"].append(d.get("speed_tg", 0))
                        configs[key]["total"].append(d.get("speed", 0))
                    except json.JSONDecodeError:
                        pass
    return configs


def get_median(configs, numa, system, metric):
    key = (numa, system)
    if key in configs and configs[key][metric]:
        return median(configs[key][metric])
    return 0.0


def plot_3way_scaling(configs, output_dir):
    """Generate three-way strong scaling comparison."""
    numas = ["1n", "4n", "8n"]
    numa_count = [1, 4, 8]

    # =====================================================================
    # Figure 1: Three-way Strong Scaling (TG + PP side by side)
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- TG (decode) ---
    # FJ Pure baseline (1N uses fj, multi-NUMA also uses fj with no TP)
    fj_tg = [get_median(configs, n, "fj", "tg") for n in numas]
    # FJ+TP (1N uses fj baseline, 4N/8N use fjtp with TP)
    fjtp_tg = [get_median(configs, "1n", "fj", "tg")]  # 1N baseline same
    fjtp_tg.append(get_median(configs, "4n", "fjtp", "tg"))
    fjtp_tg.append(get_median(configs, "8n", "fjtp", "tg"))
    # TaskInfer (1N uses tf, 4N/8N use tf with TP)
    tf_tg = [get_median(configs, n, "tf", "tg") for n in numas]

    ax1.plot(numa_count, fj_tg, "o-", color=COLOR_FJ, linewidth=2.5, markersize=8,
             label="FJ Pure (no TP)", zorder=5)
    ax1.plot(numa_count, fjtp_tg, "s-", color=COLOR_FJTP, linewidth=2.5, markersize=8,
             label="FJ + TP + row-shard", zorder=5)
    ax1.plot(numa_count, tf_tg, "D-", color=COLOR_TF, linewidth=2.5, markersize=8,
             label="TaskInfer + TP + row-shard", zorder=5)

    # Ideal line from FJ 1N baseline
    if fj_tg[0] > 0:
        ideal = [fj_tg[0] * n for n in numa_count]
        ax1.plot(numa_count, ideal, "--", color=COLOR_IDEAL, linewidth=1.5,
                 label="Ideal linear", alpha=0.6)

    for vals, color, offset in [(fj_tg, COLOR_FJ, -12), (fjtp_tg, COLOR_FJTP, 5), (tf_tg, COLOR_TF, 12)]:
        for n, v in zip(numa_count, vals):
            if v > 0:
                ax1.annotate(f"{v:.2f}", (n, v), textcoords="offset points",
                             xytext=(0, offset), ha="center", fontsize=8, fontweight="bold", color=color)

    ax1.set_xlabel("NUMA Nodes", fontsize=11)
    ax1.set_ylabel("Decode Throughput (tokens/s)", fontsize=11)
    ax1.set_title("(a) Decode (TG) Throughput", fontsize=12, fontweight="bold")
    ax1.set_xticks(numa_count)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)

    # --- PP (prefill) ---
    fj_pp = [get_median(configs, n, "fj", "pp") for n in numas]
    fjtp_pp = [get_median(configs, "1n", "fj", "pp")]
    fjtp_pp.append(get_median(configs, "4n", "fjtp", "pp"))
    fjtp_pp.append(get_median(configs, "8n", "fjtp", "pp"))
    tf_pp = [get_median(configs, n, "tf", "pp") for n in numas]

    ax2.plot(numa_count, fj_pp, "o-", color=COLOR_FJ, linewidth=2.5, markersize=8,
             label="FJ Pure (no TP)", zorder=5)
    ax2.plot(numa_count, fjtp_pp, "s-", color=COLOR_FJTP, linewidth=2.5, markersize=8,
             label="FJ + TP + row-shard", zorder=5)
    ax2.plot(numa_count, tf_pp, "D-", color=COLOR_TF, linewidth=2.5, markersize=8,
             label="TaskInfer + TP + row-shard", zorder=5)

    if fj_pp[0] > 0:
        ideal = [fj_pp[0] * n for n in numa_count]
        ax2.plot(numa_count, ideal, "--", color=COLOR_IDEAL, linewidth=1.5,
                 label="Ideal linear", alpha=0.6)

    for vals, color, offset in [(fj_pp, COLOR_FJ, -12), (fjtp_pp, COLOR_FJTP, 5), (tf_pp, COLOR_TF, 12)]:
        for n, v in zip(numa_count, vals):
            if v > 0:
                ax2.annotate(f"{v:.1f}", (n, v), textcoords="offset points",
                             xytext=(0, offset), ha="center", fontsize=8, fontweight="bold", color=color)

    ax2.set_xlabel("NUMA Nodes", fontsize=11)
    ax2.set_ylabel("Prefill Throughput (tokens/s)", fontsize=11)
    ax2.set_title("(b) Prefill (PP) Throughput", fontsize=12, fontweight="bold")
    ax2.set_xticks(numa_count)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)

    fig.suptitle("Strong Scaling: FJ Pure vs FJ+TP vs TaskInfer\n"
                 "Qwen3-4B F16 | PP=64 TG=16 | 24t/NUMA | Intel Xeon 8160",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    path = os.path.join(output_dir, "fig_3way_scaling.png")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)

    # =====================================================================
    # Figure 2: Scaling Efficiency E(N) = T(N) / [N * T(1)]
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, metric, title in [(ax1, "tg", "(a) Decode (TG) Efficiency"),
                               (ax2, "pp", "(b) Prefill (PP) Efficiency")]:
        fj_vals = [get_median(configs, n, "fj", metric) for n in numas]
        fjtp_vals = [get_median(configs, "1n", "fj", metric)]
        fjtp_vals.append(get_median(configs, "4n", "fjtp", metric))
        fjtp_vals.append(get_median(configs, "8n", "fjtp", metric))
        tf_vals = [get_median(configs, n, "tf", metric) for n in numas]

        def eff(vals):
            if vals[0] <= 0:
                return [0] * len(vals)
            return [vals[i] / (numa_count[i] * vals[0]) for i in range(len(vals))]

        fj_eff = eff(fj_vals)
        fjtp_eff = eff(fjtp_vals)
        tf_eff = eff(tf_vals)

        ax.plot(numa_count, fj_eff, "o-", color=COLOR_FJ, linewidth=2.5, markersize=8, label="FJ Pure")
        ax.plot(numa_count, fjtp_eff, "s-", color=COLOR_FJTP, linewidth=2.5, markersize=8, label="FJ + TP")
        ax.plot(numa_count, tf_eff, "D-", color=COLOR_TF, linewidth=2.5, markersize=8, label="TaskInfer + TP")
        ax.axhline(1.0, color=COLOR_IDEAL, linestyle="--", linewidth=1.5, alpha=0.7, label="Ideal (100%)")

        for vals_eff, color, offset in [(fj_eff, COLOR_FJ, -12), (fjtp_eff, COLOR_FJTP, 5), (tf_eff, COLOR_TF, 12)]:
            for n, e in zip(numa_count, vals_eff):
                ax.annotate(f"{e:.0%}", (n, e), textcoords="offset points",
                            xytext=(8, offset), fontsize=8, color=color, fontweight="bold")

        ax.set_xlabel("NUMA Nodes", fontsize=11)
        ax.set_ylabel("Scaling Efficiency E(N)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(numa_count)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Scaling Efficiency: E(N) = T(N) / [N * T(1)]\n"
                 "Qwen3-4B F16 | PP=64 TG=16 | Intel Xeon 8160",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    path = os.path.join(output_dir, "fig_3way_efficiency.png")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)

    # =====================================================================
    # Figure 3: Speedup Bar Chart (grouped bars)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(10, 5.5))

    systems = ["FJ Pure", "FJ + TP\n+ row-shard", "TaskInfer\n+ TP + row-shard"]
    x = np.arange(len(numas))
    width = 0.25

    for i, (sys_label, color, sys_key_fn) in enumerate([
        ("FJ Pure", COLOR_FJ, lambda n: get_median(configs, n, "fj", "tg")),
        ("FJ + TP", COLOR_FJTP, lambda n: get_median(configs, n, "fjtp", "tg") if n != "1n" else get_median(configs, "1n", "fj", "tg")),
        ("TaskInfer", COLOR_TF, lambda n: get_median(configs, n, "tf", "tg")),
    ]):
        vals = [sys_key_fn(n) for n in numas]
        bars = ax.bar(x + i * width - width, vals, width, color=color, alpha=0.85, label=sys_label)
        for j, v in enumerate(vals):
            if v > 0:
                ax.text(x[j] + i * width - width, v + 0.03, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold", color=color)

    ax.set_xlabel("NUMA Configuration", fontsize=11)
    ax.set_ylabel("Decode Throughput (tokens/s)", fontsize=11)
    ax.set_title("Decode (TG) Throughput: Three-Way Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}N ({n * 24}t)" for n in numa_count])
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.suptitle("Qwen3-4B F16 | PP=64 TG=16 | Intel Xeon 8160",
                 fontsize=11, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(output_dir, "fig_3way_bars.png")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)

    # =====================================================================
    # Figure 4: Speedup vs FJ Pure
    # =====================================================================
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for sys_label, color, sys_key_fn in [
        ("FJ + TP + row-shard", COLOR_FJTP,
         lambda n: get_median(configs, n, "fjtp", "tg") if n != "1n" else get_median(configs, "1n", "fj", "tg")),
        ("TaskInfer + TP + row-shard", COLOR_TF,
         lambda n: get_median(configs, n, "tf", "tg")),
    ]:
        fj_base = [get_median(configs, n, "fj", "tg") for n in numas]
        tp_vals = [sys_key_fn(n) for n in numas]
        speedups = [t / f if f > 0 else 0 for t, f in zip(tp_vals, fj_base)]
        ax.plot(numa_count, speedups, "o-", color=color, linewidth=2.5, markersize=8, label=sys_label)
        for n, s in zip(numa_count, speedups):
            ax.annotate(f"{s:.1f}x", (n, s), textcoords="offset points",
                        xytext=(8, 5), fontsize=9, color=color, fontweight="bold")

    ax.axhline(1.0, color=COLOR_IDEAL, linestyle="--", linewidth=1.5, alpha=0.7, label="FJ Pure baseline")
    ax.set_xlabel("NUMA Nodes", fontsize=11)
    ax.set_ylabel("Speedup vs FJ Pure", fontsize=11)
    ax.set_title("Speedup Over FJ Pure (Decode, TG)", fontsize=12, fontweight="bold")
    ax.set_xticks(numa_count)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig_3way_speedup.png")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "results_taskinfer")
    configs = load_results(data_dir)
    print(f"Loaded {sum(len(v['tg']) for v in configs.values())} runs across {len(configs)} configs")
    plot_3way_scaling(configs, base_dir)
    print("\nAll 3-way figures generated!")
