#!/usr/bin/env python3
"""
parse_and_plot.py — Parse llama-batched-bench + GGML_PERF_ALL data and generate
stacked bar charts showing Compute / Barrier / Idle breakdown for 3 models.

Supports the all-thread instrumentation format:
  GGML_PERF_ALL|nth=24|graph_us=...|avg_compute_us=...|avg_barrier_us=...|avg_idle_us=...
              |t0_compute_us=...|t0_barrier_us=...|t0_idle_us=...
              |max_compute_us=...|min_compute_us=...

Usage:
    python3 parse_and_plot.py [--data-dir raw_data] [--output bench_breakdown.png]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def parse_log(filepath):
    """
    Parse a single model's log file.
    Returns:
      - perf_lines: list of dicts from GGML_PERF_ALL lines
      - jsonl_lines: list of dicts from JSONL output
    """
    perf_lines = []
    jsonl_lines = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GGML_PERF_ALL|"):
                parts = line.split("|")
                rec = {}
                for p in parts[1:]:
                    k, v = p.split("=")
                    rec[k] = float(v)
                perf_lines.append(rec)
            elif line.startswith("{"):
                try:
                    data = json.loads(line)
                    jsonl_lines.append(data)
                except json.JSONDecodeError:
                    pass

    return perf_lines, jsonl_lines


def aggregate_perf(perf_lines, jsonl_lines):
    """
    Match GGML_PERF_ALL lines to (pp, tg, pl) configs from JSONL.
    Each graph compute call produces one GGML_PERF_ALL line.
    """
    results = []

    if not jsonl_lines:
        return results

    n_batch = jsonl_lines[0].get("n_batch", 512)
    is_pp_shared = jsonl_lines[0].get("is_pp_shared", 0)

    # Skip warmup: 1 graph compute for warmup
    warmup_graphs = 1
    perf_idx = warmup_graphs

    for jl in jsonl_lines:
        pp = jl["pp"]
        tg = jl["tg"]
        pl = jl["pl"]

        pp_tokens = pp if is_pp_shared else pp * pl
        pp_graphs = max(1, (pp_tokens + n_batch - 1) // n_batch)
        tg_graphs = tg
        total_graphs = pp_graphs + tg_graphs

        if perf_idx + total_graphs > len(perf_lines):
            config_perf = perf_lines[perf_idx:]
            perf_idx = len(perf_lines)
        else:
            config_perf = perf_lines[perf_idx:perf_idx + total_graphs]
            perf_idx += total_graphs

        if not config_perf:
            continue

        pp_perf = config_perf[:pp_graphs]
        tg_perf = config_perf[pp_graphs:]

        def sum_field(records, field):
            return sum(r.get(field, 0) for r in records)

        # PP aggregation (all-thread averages)
        pp_graph_us       = sum_field(pp_perf, "graph_us")
        pp_avg_compute_us = sum_field(pp_perf, "avg_compute_us")
        pp_avg_barrier_us = sum_field(pp_perf, "avg_barrier_us")
        pp_avg_idle_us    = sum_field(pp_perf, "avg_idle_us")
        pp_t0_compute_us  = sum_field(pp_perf, "t0_compute_us")
        pp_t0_barrier_us  = sum_field(pp_perf, "t0_barrier_us")
        pp_t0_idle_us     = sum_field(pp_perf, "t0_idle_us")
        pp_max_compute_us = sum_field(pp_perf, "max_compute_us")
        pp_min_compute_us = sum_field(pp_perf, "min_compute_us")

        # TG aggregation
        tg_graph_us       = sum_field(tg_perf, "graph_us")
        tg_avg_compute_us = sum_field(tg_perf, "avg_compute_us")
        tg_avg_barrier_us = sum_field(tg_perf, "avg_barrier_us")
        tg_avg_idle_us    = sum_field(tg_perf, "avg_idle_us")
        tg_t0_compute_us  = sum_field(tg_perf, "t0_compute_us")
        tg_t0_barrier_us  = sum_field(tg_perf, "t0_barrier_us")
        tg_t0_idle_us     = sum_field(tg_perf, "t0_idle_us")
        tg_max_compute_us = sum_field(tg_perf, "max_compute_us")
        tg_min_compute_us = sum_field(tg_perf, "min_compute_us")

        # Total
        total_graph_us       = pp_graph_us + tg_graph_us
        total_avg_compute_us = pp_avg_compute_us + tg_avg_compute_us
        total_avg_barrier_us = pp_avg_barrier_us + tg_avg_barrier_us
        total_avg_idle_us    = pp_avg_idle_us + tg_avg_idle_us
        total_t0_compute_us  = pp_t0_compute_us + tg_t0_compute_us
        total_t0_barrier_us  = pp_t0_barrier_us + tg_t0_barrier_us
        total_t0_idle_us     = pp_t0_idle_us + tg_t0_idle_us

        results.append({
            "pp": pp, "tg": tg, "pl": pl,
            "t_pp": jl.get("t_pp", 0), "t_tg": jl.get("t_tg", 0),
            "speed_pp": jl.get("speed_pp", 0), "speed_tg": jl.get("speed_tg", 0),
            # PP
            "pp_graph_us": pp_graph_us,
            "pp_avg_compute_us": pp_avg_compute_us,
            "pp_avg_barrier_us": pp_avg_barrier_us,
            "pp_avg_idle_us": max(0, pp_avg_idle_us),
            "pp_t0_compute_us": pp_t0_compute_us,
            "pp_t0_barrier_us": pp_t0_barrier_us,
            "pp_t0_idle_us": max(0, pp_t0_idle_us),
            "pp_max_compute_us": pp_max_compute_us,
            "pp_min_compute_us": pp_min_compute_us,
            # TG
            "tg_graph_us": tg_graph_us,
            "tg_avg_compute_us": tg_avg_compute_us,
            "tg_avg_barrier_us": tg_avg_barrier_us,
            "tg_avg_idle_us": max(0, tg_avg_idle_us),
            "tg_t0_compute_us": tg_t0_compute_us,
            "tg_t0_barrier_us": tg_t0_barrier_us,
            "tg_t0_idle_us": max(0, tg_t0_idle_us),
            "tg_max_compute_us": tg_max_compute_us,
            "tg_min_compute_us": tg_min_compute_us,
            # Total
            "total_graph_us": total_graph_us,
            "total_avg_compute_us": total_avg_compute_us,
            "total_avg_barrier_us": total_avg_barrier_us,
            "total_avg_idle_us": max(0, total_avg_idle_us),
            "total_t0_compute_us": total_t0_compute_us,
            "total_t0_barrier_us": total_t0_barrier_us,
            "total_t0_idle_us": max(0, total_t0_idle_us),
        })

    return results


MODEL_DISPLAY = {
    "qwen3_06b": "Qwen3-0.6B",
    "qwen3_4b": "Qwen3-4B",
    "llama3_8b": "Llama3-8B",
}


def plot_allthread_breakdown(all_data, output_path):
    """
    Main chart: 3×3 grid (PP/TG/Total × 3 models)
    Stacked bars using ALL-THREAD AVERAGE: Compute / Barrier / Idle
    """
    model_names = list(all_data.keys())
    n_models = len(model_names)
    if n_models == 0:
        print("No data to plot!")
        return

    fig, axes = plt.subplots(3, n_models, figsize=(6 * n_models, 14), squeeze=False)

    colors = {"Compute": "#4285F4", "Barrier": "#EA4335", "Idle": "#9AA0A6"}

    phases = [
        ("PP (Prompt Processing)", "pp"),
        ("TG (Token Generation)", "tg"),
        ("Total (PP+TG)", "total"),
    ]

    for col, model_name in enumerate(model_names):
        data = all_data[model_name]
        batch_sizes = [d["pl"] for d in data]

        for row, (phase_title, prefix) in enumerate(phases):
            ax = axes[row][col]

            graph_us   = np.array([d[f"{prefix}_graph_us"] for d in data])
            compute_us = np.array([d[f"{prefix}_avg_compute_us"] for d in data])
            barrier_us = np.array([d[f"{prefix}_avg_barrier_us"] for d in data])
            idle_us    = np.array([d[f"{prefix}_avg_idle_us"] for d in data])

            total = graph_us.copy()
            total[total == 0] = 1
            compute_pct = compute_us / total * 100
            barrier_pct = barrier_us / total * 100
            idle_pct    = idle_us / total * 100

            x = np.arange(len(batch_sizes))
            width = 0.6

            ax.bar(x, compute_pct, width, label="Compute", color=colors["Compute"], alpha=0.9)
            ax.bar(x, barrier_pct, width, bottom=compute_pct, label="Barrier", color=colors["Barrier"], alpha=0.9)
            ax.bar(x, idle_pct, width, bottom=compute_pct + barrier_pct, label="Idle", color=colors["Idle"], alpha=0.9)

            # Labels on barrier and idle segments
            for i in range(len(batch_sizes)):
                if barrier_pct[i] > 2:
                    ax.text(x[i], compute_pct[i] + barrier_pct[i] / 2,
                            f"{barrier_pct[i]:.1f}%", ha="center", va="center",
                            fontsize=7, fontweight="bold", color="white")
                if idle_pct[i] > 2:
                    ax.text(x[i], compute_pct[i] + barrier_pct[i] + idle_pct[i] / 2,
                            f"{idle_pct[i]:.1f}%", ha="center", va="center",
                            fontsize=7, fontweight="bold", color="black")

            ax.set_ylim(0, 105)
            ax.set_xticks(x)
            ax.set_xticklabels(batch_sizes)
            ax.set_ylabel("Time proportion (%)")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.set_axisbelow(True)

            if row == 0:
                display_name = MODEL_DISPLAY.get(model_name, model_name)
                ax.set_title(f"{display_name}\n{phase_title}", fontsize=12, fontweight="bold")
            else:
                ax.set_title(phase_title, fontsize=11)
            if row == 2:
                ax.set_xlabel("Batch size (parallel sequences)")
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "CPU Time Breakdown (All-Thread Average): Compute / Barrier / Idle\n"
        "1NUMA (node 1) · 24 threads · PP64 TG32 · Intel Xeon 8160",
        fontsize=14, fontweight="bold", y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_t0_vs_avg(all_data, output_path):
    """
    Comparison chart: Thread-0 vs All-Thread Average breakdown (Total only).
    Two bars per batch size: left = thread-0, right = avg-all-threads.
    This shows how thread-0 (busiest) differs from the average worker.
    """
    model_names = list(all_data.keys())
    n_models = len(model_names)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5), squeeze=False)

    c_compute = "#4285F4"
    c_barrier = "#EA4335"
    c_idle = "#9AA0A6"

    for col, model_name in enumerate(model_names):
        data = all_data[model_name]
        batch_sizes = [d["pl"] for d in data]
        ax = axes[0][col]

        n = len(batch_sizes)
        x = np.arange(n)
        width = 0.35

        # Thread-0
        t0_graph   = np.array([d["total_graph_us"] for d in data])
        t0_compute = np.array([d["total_t0_compute_us"] for d in data])
        t0_barrier = np.array([d["total_t0_barrier_us"] for d in data])
        t0_idle    = np.array([d["total_t0_idle_us"] for d in data])

        t0_total = t0_graph.copy(); t0_total[t0_total == 0] = 1
        t0_c_pct = t0_compute / t0_total * 100
        t0_b_pct = t0_barrier / t0_total * 100
        t0_i_pct = t0_idle / t0_total * 100

        # All-thread average
        avg_compute = np.array([d["total_avg_compute_us"] for d in data])
        avg_barrier = np.array([d["total_avg_barrier_us"] for d in data])
        avg_idle    = np.array([d["total_avg_idle_us"] for d in data])

        avg_total = t0_graph.copy(); avg_total[avg_total == 0] = 1
        avg_c_pct = avg_compute / avg_total * 100
        avg_b_pct = avg_barrier / avg_total * 100
        avg_i_pct = avg_idle / avg_total * 100

        # Thread-0 bars (left)
        ax.bar(x - width/2, t0_c_pct, width, color=c_compute, alpha=0.9)
        ax.bar(x - width/2, t0_b_pct, width, bottom=t0_c_pct, color=c_barrier, alpha=0.9)
        ax.bar(x - width/2, t0_i_pct, width, bottom=t0_c_pct + t0_b_pct, color=c_idle, alpha=0.9)

        # Avg bars (right)
        ax.bar(x + width/2, avg_c_pct, width, color=c_compute, alpha=0.5, edgecolor=c_compute, linewidth=0.5)
        ax.bar(x + width/2, avg_b_pct, width, bottom=avg_c_pct, color=c_barrier, alpha=0.5, edgecolor=c_barrier, linewidth=0.5)
        ax.bar(x + width/2, avg_i_pct, width, bottom=avg_c_pct + avg_b_pct, color=c_idle, alpha=0.5, edgecolor=c_idle, linewidth=0.5)

        # Barrier % labels
        for i in range(n):
            if t0_b_pct[i] > 2:
                ax.text(x[i] - width/2, t0_c_pct[i] + t0_b_pct[i]/2,
                        f"{t0_b_pct[i]:.0f}%", ha="center", va="center", fontsize=6, color="white", fontweight="bold")
            if avg_b_pct[i] > 2:
                ax.text(x[i] + width/2, avg_c_pct[i] + avg_b_pct[i]/2,
                        f"{avg_b_pct[i]:.0f}%", ha="center", va="center", fontsize=6, color="white", fontweight="bold")

        ax.set_ylim(0, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes)
        ax.set_ylabel("Time proportion (%)")
        ax.set_xlabel("Batch size")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        display_name = MODEL_DISPLAY.get(model_name, model_name)
        ax.set_title(display_name, fontsize=12, fontweight="bold")

        if col == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=c_compute, alpha=0.9, label="Compute (T0)"),
                Patch(facecolor=c_barrier, alpha=0.9, label="Barrier (T0)"),
                Patch(facecolor=c_idle, alpha=0.9, label="Idle (T0)"),
                Patch(facecolor=c_compute, alpha=0.5, label="Compute (Avg)"),
                Patch(facecolor=c_barrier, alpha=0.5, label="Barrier (Avg)"),
                Patch(facecolor=c_idle, alpha=0.5, label="Idle (Avg)"),
            ]
            ax.legend(handles=legend_elements, fontsize=7, loc="upper right", ncol=2)

    fig.suptitle(
        "Thread-0 (solid) vs All-Thread Average (faded) — Total PP64+TG32\n"
        "1NUMA (node 1) · 24 threads · Intel Xeon 8160",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_absolute(all_data, output_path):
    """
    Absolute time (ms) stacked bar: Compute / Barrier / Idle (all-thread avg).
    """
    model_names = list(all_data.keys())
    n_models = len(model_names)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5), squeeze=False)

    colors = {"Compute": "#4285F4", "Barrier": "#EA4335", "Idle": "#9AA0A6"}

    for col, model_name in enumerate(model_names):
        data = all_data[model_name]
        batch_sizes = [d["pl"] for d in data]
        ax = axes[0][col]

        compute_ms = np.array([d["total_avg_compute_us"] / 1000 for d in data])
        barrier_ms = np.array([d["total_avg_barrier_us"] / 1000 for d in data])
        idle_ms    = np.array([d["total_avg_idle_us"] / 1000 for d in data])

        x = np.arange(len(batch_sizes))
        width = 0.6

        ax.bar(x, compute_ms, width, label="Compute", color=colors["Compute"], alpha=0.9)
        ax.bar(x, barrier_ms, width, bottom=compute_ms, label="Barrier", color=colors["Barrier"], alpha=0.9)
        ax.bar(x, idle_ms, width, bottom=compute_ms + barrier_ms, label="Idle", color=colors["Idle"], alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes)
        ax.set_ylabel("Time (ms)")
        ax.set_xlabel("Batch size")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        display_name = MODEL_DISPLAY.get(model_name, model_name)
        ax.set_title(display_name, fontsize=12, fontweight="bold")

        if col == 0:
            ax.legend(fontsize=9)

    fig.suptitle(
        "CPU Time Breakdown (Absolute, All-Thread Avg) — Total = PP64 + TG32\n"
        "1NUMA (node 1) · 24 threads · Intel Xeon 8160",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_compute_imbalance(all_data, output_path):
    """
    Show max_compute vs min_compute per config (TG only) to visualize load imbalance.
    """
    model_names = list(all_data.keys())
    n_models = len(model_names)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)

    for col, model_name in enumerate(model_names):
        data = all_data[model_name]
        batch_sizes = [d["pl"] for d in data]
        ax = axes[0][col]

        x = np.arange(len(batch_sizes))
        width = 0.6

        # TG phase: per-graph-call averages → compute imbalance
        tg_max = np.array([d["tg_max_compute_us"] / 1000 for d in data])
        tg_min = np.array([d["tg_min_compute_us"] / 1000 for d in data])
        tg_avg = np.array([d["tg_avg_compute_us"] / 1000 for d in data])

        ax.bar(x, tg_min, width, label="Min thread compute", color="#34A853", alpha=0.8)
        ax.bar(x, tg_max - tg_min, width, bottom=tg_min,
               label="Imbalance (max−min)", color="#FBBC04", alpha=0.8)

        # Mark avg line
        for i in range(len(batch_sizes)):
            ax.plot([x[i] - width/2, x[i] + width/2], [tg_avg[i], tg_avg[i]],
                    color="black", linewidth=1.5, zorder=5)
            imb_pct = (tg_max[i] - tg_min[i]) / max(tg_max[i], 0.001) * 100
            ax.text(x[i], tg_max[i] + tg_max[i]*0.02,
                    f"{imb_pct:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes)
        ax.set_ylabel("Compute time per graph (ms) — TG phase")
        ax.set_xlabel("Batch size")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        display_name = MODEL_DISPLAY.get(model_name, model_name)
        ax.set_title(display_name, fontsize=12, fontweight="bold")

        if col == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor="#34A853", alpha=0.8, label="Min thread compute"),
                plt.Rectangle((0,0),1,1, facecolor="#FBBC04", alpha=0.8, label="Imbalance (max−min)"),
                Line2D([0],[0], color="black", linewidth=1.5, label="Avg compute"),
            ]
            ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    fig.suptitle(
        "Compute Load Imbalance Across Threads — TG Phase\n"
        "1NUMA (node 1) · 24 threads · PP64 TG32 · Intel Xeon 8160",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def print_summary_table(all_data):
    """Print a detailed markdown summary table with all-thread averages."""
    print("\n## All-Thread Average Breakdown\n")
    print("| Model | Batch | Phase | Compute% | Barrier% | Idle% | Graph(ms) |")
    print("|-------|-------|-------|----------|----------|-------|-----------|")

    for model_name, data in all_data.items():
        display = MODEL_DISPLAY.get(model_name, model_name)
        for d in data:
            pl = d["pl"]
            for prefix, phase in [("pp", "PP"), ("tg", "TG"), ("total", "Tot")]:
                g = d[f"{prefix}_graph_us"]
                if g == 0:
                    continue
                c_pct = d[f"{prefix}_avg_compute_us"] / g * 100
                b_pct = d[f"{prefix}_avg_barrier_us"] / g * 100
                i_pct = d[f"{prefix}_avg_idle_us"] / g * 100
                g_ms = g / 1000
                print(f"| {display} | {pl} | {phase} | {c_pct:.1f}% | {b_pct:.1f}% | {i_pct:.1f}% | {g_ms:.1f} |")

    print("\n## Thread-0 vs Avg Comparison (Total)\n")
    print("| Model | Batch | T0 Compute% | T0 Barrier% | Avg Compute% | Avg Barrier% | Imbalance(max-min ms) |")
    print("|-------|-------|-------------|-------------|--------------|--------------|----------------------|")

    for model_name, data in all_data.items():
        display = MODEL_DISPLAY.get(model_name, model_name)
        for d in data:
            pl = d["pl"]
            g = d["total_graph_us"]
            if g == 0:
                continue
            t0_c = d["total_t0_compute_us"] / g * 100
            t0_b = d["total_t0_barrier_us"] / g * 100
            avg_c = d["total_avg_compute_us"] / g * 100
            avg_b = d["total_avg_barrier_us"] / g * 100
            imb_ms = (d["total_graph_us"] - d["total_avg_compute_us"] - d["total_avg_barrier_us"]) / 1000
            # Max-min imbalance not directly in total, use TG as proxy
            tg_imb = (d["tg_max_compute_us"] - d["tg_min_compute_us"]) / 1000
            print(f"| {display} | {pl} | {t0_c:.1f}% | {t0_b:.1f}% | {avg_c:.1f}% | {avg_b:.1f}% | {tg_imb:.1f} |")


def main():
    parser = argparse.ArgumentParser(description="Parse and plot benchmark breakdown (all-thread)")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__) or ".", "raw_data"),
                        help="Directory containing raw benchmark logs")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__) or ".", "bench_breakdown.png"),
                        help="Output image path")
    args = parser.parse_args()

    model_order = ["qwen3_06b", "qwen3_4b", "llama3_8b"]
    all_data = {}

    for name in model_order:
        logfile = os.path.join(args.data_dir, f"{name}.log")
        if not os.path.exists(logfile):
            print(f"Warning: {logfile} not found, skipping")
            continue

        perf_lines, jsonl_lines = parse_log(logfile)
        print(f"{name}: {len(perf_lines)} GGML_PERF_ALL lines, {len(jsonl_lines)} JSONL lines")

        results = aggregate_perf(perf_lines, jsonl_lines)
        if results:
            all_data[name] = results

    if not all_data:
        print("No data found. Run benchmarks first!")
        sys.exit(1)

    print_summary_table(all_data)

    # Plot 1: Percentage breakdown (3×3: PP/TG/Total × 3 models) — all-thread avg
    plot_allthread_breakdown(all_data, args.output)

    # Plot 2: Absolute time
    abs_output = args.output.replace(".png", "_absolute.png")
    plot_absolute(all_data, abs_output)

    # Plot 3: Thread-0 vs Average comparison
    t0_output = args.output.replace(".png", "_t0_vs_avg.png")
    plot_t0_vs_avg(all_data, t0_output)

    # Plot 4: Compute imbalance
    imb_output = args.output.replace(".png", "_imbalance.png")
    plot_compute_imbalance(all_data, imb_output)


if __name__ == "__main__":
    main()
