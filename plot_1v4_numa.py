#!/usr/bin/env python3
"""
plot_1v4_numa.py — Compare 1NUMA vs 4NUMA per-operator time breakdown.
Reads per-batch log files from raw_data/ (1NUMA) and raw_data_4numa/ (4NUMA).

Generates:
  1. Per-graph-call time comparison (PP+TG combined): stacked Compute + Barrier
  2. Breakdown % comparison
  3. Compute & Barrier absolute comparison
"""

import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


MODEL_DISPLAY = {
    "qwen3_06b": "Qwen3-0.6B",
    "qwen3_4b": "Qwen3-4B",
    "llama3_8b": "Llama3-8B",
}

MODEL_ORDER = ["qwen3_06b", "qwen3_4b", "llama3_8b"]
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
BASE_DIR = os.path.dirname(__file__) or "."


def parse_tg_perf(filepath):
    """Parse a per-batch log file. Return average per-graph-call stats for ALL ops (PP+TG combined, skip warmup)."""
    perf_lines = []
    jsonl_data = None

    with open(filepath) as f:
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
                    jsonl_data = json.loads(line)
                except json.JSONDecodeError:
                    pass

    # Skip warmup (first 1 perf line), use ALL remaining (PP + TG combined)
    all_perf = perf_lines[1:] if len(perf_lines) > 1 else perf_lines

    n = len(all_perf)
    if n == 0:
        return None

    avg = {}
    for key in ["graph_us", "avg_compute_us", "avg_barrier_us", "avg_idle_us",
                "max_compute_us", "min_compute_us"]:
        avg[key] = sum(p.get(key, 0) for p in all_perf) / n

    # Also store total wall time (sum of all graph_us)
    avg["total_graph_us"] = sum(p.get("graph_us", 0) for p in all_perf)
    avg["total_compute_us"] = sum(p.get("avg_compute_us", 0) for p in all_perf)
    avg["total_barrier_us"] = sum(p.get("avg_barrier_us", 0) for p in all_perf)
    avg["n_graph_calls"] = n

    if jsonl_data:
        avg["speed_tg"] = jsonl_data.get("speed_tg", 0)
        avg["speed_pp"] = jsonl_data.get("speed_pp", 0)
        avg["t_tg"] = jsonl_data.get("t_tg", 0)
        avg["t_pp"] = jsonl_data.get("t_pp", 0)
        avg["t"] = jsonl_data.get("t", 0)
        avg["t_tg"] = jsonl_data.get("t_tg", 0)

    return avg


def load_all(data_dir):
    """Load per-batch TG stats for all models."""
    result = {}
    for model in MODEL_ORDER:
        model_data = []
        for pl in BATCH_SIZES:
            f = os.path.join(data_dir, f"{model}_pl{pl}.log")
            if not os.path.exists(f):
                model_data.append(None)
                continue
            model_data.append(parse_tg_perf(f))
        result[model] = model_data
    return result


def plot_per_op_time(data_1n, data_4n, output_path):
    """
    Chart 1: Per-graph-call time (ms), PP+TG combined.
    Two bars per batch: left=1NUMA, right=4NUMA. Stacked: Compute + Barrier.
    """
    n_models = len(MODEL_ORDER)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), squeeze=False)

    c_compute = "#4285F4"
    c_barrier = "#EA4335"

    for col, model in enumerate(MODEL_ORDER):
        ax = axes[0][col]
        d1 = data_1n[model]
        d4 = data_4n[model]

        n = len(BATCH_SIZES)
        x = np.arange(n)
        width = 0.35

        c1 = np.array([d["avg_compute_us"] / 1000 if d else 0 for d in d1])
        b1 = np.array([d["avg_barrier_us"] / 1000 if d else 0 for d in d1])
        c4 = np.array([d["avg_compute_us"] / 1000 if d else 0 for d in d4])
        b4 = np.array([d["avg_barrier_us"] / 1000 if d else 0 for d in d4])

        # 1NUMA bars (left)
        ax.bar(x - width/2, c1, width, color=c_compute, alpha=0.9, label="Compute (1N)")
        ax.bar(x - width/2, b1, width, bottom=c1, color=c_barrier, alpha=0.9, label="Barrier (1N)")

        # 4NUMA bars (right)
        ax.bar(x + width/2, c4, width, color=c_compute, alpha=0.45, edgecolor=c_compute,
               linewidth=1, label="Compute (8N)")
        ax.bar(x + width/2, b4, width, bottom=c4, color=c_barrier, alpha=0.45, edgecolor=c_barrier,
               linewidth=1, label="Barrier (8N)")

        # Ratio labels on top
        for i in range(n):
            total_1n = c1[i] + b1[i]
            total_4n = c4[i] + b4[i]
            if total_1n > 0:
                ratio = total_4n / total_1n
                ax.text(x[i], max(total_1n, total_4n) * 1.02,
                        f"{ratio:.1f}x", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#D32F2F")

        ax.set_xticks(x)
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_ylabel("Time per graph call (ms)")
        ax.set_xlabel("Batch size")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=12, fontweight="bold")

        if col == 0:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Per-Operator Time: 1NUMA (24t) vs 8NUMA (192t) — PP64+TG64\n"
        "Stacked: Compute + Barrier · Intel Xeon 8160",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_breakdown_compare(data_1n, data_4n, output_path):
    """
    Chart 2: Side-by-side percentage breakdown.
    Two bars per batch: left=1NUMA %, right=4NUMA %.
    """
    n_models = len(MODEL_ORDER)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5), squeeze=False)

    c_compute = "#4285F4"
    c_barrier = "#EA4335"
    c_idle = "#9AA0A6"

    for col, model in enumerate(MODEL_ORDER):
        ax = axes[0][col]
        d1 = data_1n[model]
        d4 = data_4n[model]

        n = len(BATCH_SIZES)
        x = np.arange(n)
        width = 0.35

        def pcts(data_list):
            cp, bp, ip = [], [], []
            for d in data_list:
                if d and d["graph_us"] > 0:
                    g = d["graph_us"]
                    cp.append(d["avg_compute_us"] / g * 100)
                    bp.append(d["avg_barrier_us"] / g * 100)
                    ip.append(d.get("avg_idle_us", 0) / g * 100)
                else:
                    cp.append(0); bp.append(0); ip.append(0)
            return np.array(cp), np.array(bp), np.array(ip)

        c1p, b1p, i1p = pcts(d1)
        c4p, b4p, i4p = pcts(d4)

        # 1NUMA (left, solid)
        ax.bar(x - width/2, c1p, width, color=c_compute, alpha=0.9)
        ax.bar(x - width/2, b1p, width, bottom=c1p, color=c_barrier, alpha=0.9)

        # 4NUMA (right, faded)
        ax.bar(x + width/2, c4p, width, color=c_compute, alpha=0.45, edgecolor=c_compute, linewidth=1)
        ax.bar(x + width/2, b4p, width, bottom=c4p, color=c_barrier, alpha=0.45, edgecolor=c_barrier, linewidth=1)

        # Barrier % labels
        for i in range(n):
            if b1p[i] > 1.5:
                ax.text(x[i] - width/2, c1p[i] + b1p[i]/2,
                        f"{b1p[i]:.0f}%", ha="center", va="center", fontsize=7,
                        fontweight="bold", color="white")
            if b4p[i] > 1.5:
                ax.text(x[i] + width/2, c4p[i] + b4p[i]/2,
                        f"{b4p[i]:.0f}%", ha="center", va="center", fontsize=7,
                        fontweight="bold", color="white")

        ax.set_ylim(0, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_ylabel("Time proportion (%)")
        ax.set_xlabel("Batch size")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=12, fontweight="bold")

        if col == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=c_compute, alpha=0.9, label="Compute (1N)"),
                Patch(facecolor=c_barrier, alpha=0.9, label="Barrier (1N)"),
                Patch(facecolor=c_compute, alpha=0.45, label="Compute (8N)"),
                Patch(facecolor=c_barrier, alpha=0.45, label="Barrier (8N)"),
            ]
            ax.legend(handles=legend_elements, fontsize=7, loc="lower right", ncol=2)

    fig.suptitle(
        "Barrier% Comparison: 1NUMA (solid) vs 4NUMA (faded) — PP64+TG64\n"
        "Intel Xeon 8160 · PP64 TG32",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_compute_and_barrier_abs(data_1n, data_4n, output_path):
    """
    Chart 3: Absolute compute time + barrier time side by side.
    Shows that 4NUMA compute is SLOWER than 1NUMA despite 4x cores.
    """
    n_models = len(MODEL_ORDER)
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10), squeeze=False)

    for col, model in enumerate(MODEL_ORDER):
        d1 = data_1n[model]
        d4 = data_4n[model]
        n = len(BATCH_SIZES)
        x = np.arange(n)
        width = 0.35

        c1 = np.array([d["avg_compute_us"] / 1000 if d else 0 for d in d1])
        c4 = np.array([d["avg_compute_us"] / 1000 if d else 0 for d in d4])
        b1 = np.array([d["avg_barrier_us"] / 1000 if d else 0 for d in d1])
        b4 = np.array([d["avg_barrier_us"] / 1000 if d else 0 for d in d4])

        # Row 0: Compute time
        ax = axes[0][col]
        bars1 = ax.bar(x - width/2, c1, width, color="#4285F4", alpha=0.9, label="1NUMA (24t)")
        bars4 = ax.bar(x + width/2, c4, width, color="#4285F4", alpha=0.45, edgecolor="#4285F4",
                       linewidth=1, label="8NUMA (192t)")
        for i in range(n):
            if c1[i] > 0:
                ratio = c4[i] / c1[i]
                ax.text(x[i], max(c1[i], c4[i]) * 1.02,
                        f"{ratio:.1f}x", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#D32F2F" if ratio > 1 else "#388E3C")

        ax.set_xticks(x)
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_ylabel("Avg compute per op (ms)")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(f"{MODEL_DISPLAY.get(model, model)}\nCompute Time", fontsize=12, fontweight="bold")
        if col == 0:
            ax.legend(fontsize=9)

        # Row 1: Barrier time
        ax = axes[1][col]
        ax.bar(x - width/2, b1, width, color="#EA4335", alpha=0.9, label="1NUMA (24t)")
        ax.bar(x + width/2, b4, width, color="#EA4335", alpha=0.45, edgecolor="#EA4335",
               linewidth=1, label="8NUMA (192t)")
        for i in range(n):
            if b1[i] > 0:
                ratio = b4[i] / b1[i]
                ax.text(x[i], max(b1[i], b4[i]) * 1.02,
                        f"{ratio:.1f}x", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#D32F2F")

        ax.set_xticks(x)
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_ylabel("Avg barrier per op (ms)")
        ax.set_xlabel("Batch size")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title("Barrier Time", fontsize=11)
        if col == 0:
            ax.legend(fontsize=9)

    fig.suptitle(
        "1NUMA (24t) vs 8NUMA (192t): Per-Operator Compute & Barrier — PP64+TG64\n"
        "Red ratio = 8NUMA slower · Intel Xeon 8160",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_e2e_walltime(data_1n, data_4n, output_path):
    """
    Chart 4: Total wall time (sum of all graph_us) for the full PP64+TG64 run.
    Shows whether 4NUMA is actually faster or slower end-to-end.
    """
    n_models = len(MODEL_ORDER)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5), squeeze=False)

    for col, model in enumerate(MODEL_ORDER):
        d1 = data_1n[model]
        d4 = data_4n[model]
        n = len(BATCH_SIZES)
        x = np.arange(n)
        width = 0.35

        # Total wall time in seconds
        t1 = np.array([d.get("total_graph_us", 0) / 1e6 if d else 0 for d in d1])
        t4 = np.array([d.get("total_graph_us", 0) / 1e6 if d else 0 for d in d4])

        ax = axes[0][col]
        ax.bar(x - width/2, t1, width, color="#4285F4", alpha=0.9, label="1NUMA (24t)")
        ax.bar(x + width/2, t4, width, color="#EA4335", alpha=0.7, label="8NUMA (192t)")

        for i in range(n):
            if t1[i] > 0:
                ratio = t4[i] / t1[i]
                color = "#D32F2F" if ratio > 1 else "#388E3C"
                label = f"{ratio:.1f}x" if ratio > 1 else f"{ratio:.2f}x"
                ax.text(x[i], max(t1[i], t4[i]) * 1.02,
                        label, ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_ylabel("Total wall time (seconds)")
        ax.set_xlabel("Batch size")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=12, fontweight="bold")
        if col == 0:
            ax.legend(fontsize=9)

    fig.suptitle(
        "End-to-End Wall Time: 1NUMA (24t) vs 8NUMA (192t) — PP64+TG64\n"
        "Red = 8NUMA slower · Intel Xeon 8160",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    dir_1n = os.path.join(BASE_DIR, "raw_data_pp64tg64")
    dir_4n = os.path.join(BASE_DIR, "raw_data_pp64tg64_8numa")

    data_1n = load_all(dir_1n)
    data_4n = load_all(dir_4n)

    # Chart 1: Per-op time stacked
    plot_per_op_time(data_1n, data_4n,
                     os.path.join(BASE_DIR, "compare_1v8_per_op.png"))

    # Chart 2: Barrier% comparison
    plot_breakdown_compare(data_1n, data_4n,
                           os.path.join(BASE_DIR, "compare_1v8_barrier_pct.png"))

    # Chart 3: Compute + Barrier absolute split
    plot_compute_and_barrier_abs(data_1n, data_4n,
                                 os.path.join(BASE_DIR, "compare_1v8_compute_barrier.png"))

    # Chart 4: E2E wall time
    plot_e2e_walltime(data_1n, data_4n,
                      os.path.join(BASE_DIR, "compare_1v8_e2e.png"))


if __name__ == "__main__":
    main()
