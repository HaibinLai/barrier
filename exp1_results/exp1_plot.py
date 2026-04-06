#!/usr/bin/env python3
"""
exp1_plot.py — Generate figures for Experiment 1: Overall Performance.

Generates:
  Figure 1: Decode throughput (TG t/s) — grouped bar chart
  Figure 2: Prefill throughput (PP t/s) — grouped bar chart
  Figure 3: Batched decode throughput — line chart across batch sizes
  Figure 4: Speedup over Fork-Join Pure
"""

import os
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = os.path.dirname(__file__)
CSV_FILE = os.path.join(BASE_DIR, "exp1_summary.csv")

# ── Style ──
SYSTEM_COLORS = {
    "fj":   "#d62728",   # red — Fork-Join Pure
    "fjtp": "#1f77b4",   # blue — Fork-Join + TP
    "task": "#2ca02c",   # green — TaskInfer
}
SYSTEM_LABELS = {
    "fj":   "Fork-Join (Pure)",
    "fjtp": "Fork-Join + TP",
    "task": "TaskInfer (Ours)",
}
MODEL_LABELS = {
    "qwen3_4b":  "Qwen3-4B",
    "llama3_8b": "LLaMA3-8B",
    "qwen25_14b": "Qwen2.5-14B",
}
NUMA_LABELS = {
    "1n": "1 NUMA (24t)",
    "4n": "4 NUMA (96t)",
    "8n": "8 NUMA (192t)",
}


def load_data():
    """Load CSV and return list of dicts."""
    with open(CSV_FILE) as f:
        return list(csv.DictReader(f))


def get_value(data, model, numa, system, stage, batch, metric="speed_tg_median"):
    """Get a specific metric value from data."""
    for row in data:
        if (row["model"] == model and row["numa"] == numa and
            row["system"] == system and row["stage"] == stage and
            int(row["batch"]) == batch):
            return float(row[metric])
    return 0.0


def plot_decode_throughput(data, output_path):
    """
    Figure 1: Decode throughput (TG t/s) — B=1 decode.
    Grouped bar chart: x-axis = model, separate subplots for each NUMA config.
    3 bars per model: FJ Pure, FJ+TP, TaskInfer
    """
    models = ["qwen3_4b", "llama3_8b"]
    numas = ["1n", "4n", "8n"]
    systems_1n = ["fj", "task"]  # No FJ+TP for 1N (TP=1)
    systems_multi = ["fj", "fjtp", "task"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for idx, numa in enumerate(numas):
        ax = axes[idx]
        systems = systems_1n if numa == "1n" else systems_multi
        n_sys = len(systems)
        n_models = len(models)
        width = 0.8 / n_sys
        x = np.arange(n_models)

        for i, sys_name in enumerate(systems):
            vals = []
            for model in models:
                v = get_value(data, model, numa, sys_name, "decode", 1, "speed_tg_median")
                vals.append(v)

            offset = (i - (n_sys - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width * 0.9,
                         color=SYSTEM_COLORS[sys_name], alpha=0.85,
                         label=SYSTEM_LABELS[sys_name])

            # Value labels
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                           f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(NUMA_LABELS[numa], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=10)
        ax.set_ylabel("Decode Throughput (tokens/s)" if idx == 0 else "", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        if idx == 2:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Experiment 1: Decode Stage Performance (batch=1, TG tokens=64)\n"
                 "Intel Xeon Platinum 8160 · 8×NUMA · 24 cores/socket",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_prefill_throughput(data, output_path):
    """
    Figure 2: Prefill throughput (PP t/s) at different prompt lengths.
    """
    models = ["qwen3_4b", "llama3_8b"]
    numas = ["1n", "4n", "8n"]
    pp_lengths = [128, 256, 512]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for idx, numa in enumerate(numas):
        ax = axes[idx]
        systems = ["fj", "task"] if numa == "1n" else ["fj", "fjtp", "task"]

        # For simplicity, use first model and show PP lengths on x-axis
        # Actually, let's show models as groups, PP=128 (representative)
        n_sys = len(systems)
        n_models = len(models)
        width = 0.8 / n_sys
        x = np.arange(n_models)

        for i, sys_name in enumerate(systems):
            vals = []
            for model in models:
                # Use PP=128 as representative
                v = get_value(data, model, numa, sys_name, "prefill", 1, "speed_pp_median")
                vals.append(v)

            offset = (i - (n_sys - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width * 0.9,
                         color=SYSTEM_COLORS[sys_name], alpha=0.85,
                         label=SYSTEM_LABELS[sys_name])

            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                           f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(NUMA_LABELS[numa], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=10)
        ax.set_ylabel("Prefill Throughput (tokens/s)" if idx == 0 else "", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        if idx == 2:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Experiment 1: Prefill Stage Performance (PP=128-512, batch=1)\n"
                 "Intel Xeon Platinum 8160 · 8×NUMA · 24 cores/socket",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_batched_decode(data, output_path):
    """
    Figure 3: Batched decode — TG throughput vs batch size.
    Line chart with separate panels per NUMA config.
    """
    models = ["qwen3_4b", "llama3_8b"]
    numas = ["4n", "8n"]  # Skip 1N for batched (less interesting)
    batches = [1, 4, 8, 16]

    fig, axes = plt.subplots(len(models), len(numas), figsize=(14, 10))

    for mi, model in enumerate(models):
        for ni, numa in enumerate(numas):
            ax = axes[mi][ni] if len(models) > 1 else axes[ni]
            systems = ["fj", "fjtp", "task"]

            for sys_name in systems:
                vals = []
                for b in batches:
                    v = get_value(data, model, numa, sys_name, "batch", b, "speed_tg_median")
                    vals.append(v)

                ax.plot(batches, vals, "o-", color=SYSTEM_COLORS[sys_name],
                       linewidth=2, markersize=6, label=SYSTEM_LABELS[sys_name])

                # Value labels
                for b, v in zip(batches, vals):
                    if v > 0:
                        ax.annotate(f"{v:.1f}", (b, v), textcoords="offset points",
                                   xytext=(0, 8), ha="center", fontsize=7)

            ax.set_title(f"{MODEL_LABELS[model]} — {NUMA_LABELS[numa]}",
                        fontsize=11, fontweight="bold")
            ax.set_xlabel("Batch Size", fontsize=10)
            ax.set_ylabel("TG Throughput (tokens/s)" if ni == 0 else "", fontsize=10)
            ax.set_xticks(batches)
            ax.grid(linestyle="--", alpha=0.3)
            ax.set_axisbelow(True)
            if mi == 0 and ni == len(numas) - 1:
                ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Experiment 1: Batched Decode Throughput\n"
                 "Intel Xeon Platinum 8160 · TG tokens=32",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_speedup(data, output_path):
    """
    Figure 4: Speedup of TaskInfer over Fork-Join Pure and FJ+TP.
    Shows both decode and prefill speedups across NUMA configs.
    """
    models = ["qwen3_4b", "llama3_8b"]
    numas = ["4n", "8n"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Decode speedup
    ax = axes[0]
    x = np.arange(len(models) * len(numas))
    labels = []
    speedup_vs_fj = []
    speedup_vs_fjtp = []

    for model in models:
        for numa in numas:
            label = f"{MODEL_LABELS[model]}\n{NUMA_LABELS[numa]}"
            labels.append(label)

            fj_val = get_value(data, model, numa, "fj", "decode", 1, "speed_tg_median")
            fjtp_val = get_value(data, model, numa, "fjtp", "decode", 1, "speed_tg_median")
            task_val = get_value(data, model, numa, "task", "decode", 1, "speed_tg_median")

            speedup_vs_fj.append(task_val / fj_val if fj_val > 0 else 0)
            speedup_vs_fjtp.append(task_val / fjtp_val if fjtp_val > 0 else 0)

    width = 0.35
    bars1 = ax.bar(x - width/2, speedup_vs_fj, width, color=SYSTEM_COLORS["fj"],
                   alpha=0.85, label="vs Fork-Join Pure")
    bars2 = ax.bar(x + width/2, speedup_vs_fjtp, width, color=SYSTEM_COLORS["fjtp"],
                   alpha=0.85, label="vs Fork-Join+TP")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                   f"{h:.2f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Decode (TG) Speedup — batch=1", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Speedup (×)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Prefill speedup
    ax = axes[1]
    speedup_pp_fj = []
    speedup_pp_fjtp = []

    for model in models:
        for numa in numas:
            fj_val = get_value(data, model, numa, "fj", "prefill", 1, "speed_pp_median")
            fjtp_val = get_value(data, model, numa, "fjtp", "prefill", 1, "speed_pp_median")
            task_val = get_value(data, model, numa, "task", "prefill", 1, "speed_pp_median")

            speedup_pp_fj.append(task_val / fj_val if fj_val > 0 else 0)
            speedup_pp_fjtp.append(task_val / fjtp_val if fjtp_val > 0 else 0)

    bars1 = ax.bar(x - width/2, speedup_pp_fj, width, color=SYSTEM_COLORS["fj"],
                   alpha=0.85, label="vs Fork-Join Pure")
    bars2 = ax.bar(x + width/2, speedup_pp_fjtp, width, color=SYSTEM_COLORS["fjtp"],
                   alpha=0.85, label="vs Fork-Join+TP")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                   f"{h:.2f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Prefill (PP) Speedup — batch=1", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Speedup (×)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.suptitle("TaskInfer Speedup over Baselines\n"
                 "Intel Xeon Platinum 8160 · >1.0 = TaskInfer is faster",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: {CSV_FILE} not found. Run exp1_parse.py first.")
        return

    data = load_data()
    print(f"Loaded {len(data)} data points from {CSV_FILE}")

    plot_decode_throughput(data, os.path.join(BASE_DIR, "exp1_decode_throughput.png"))
    plot_prefill_throughput(data, os.path.join(BASE_DIR, "exp1_prefill_throughput.png"))
    plot_batched_decode(data, os.path.join(BASE_DIR, "exp1_batched_decode.png"))
    plot_speedup(data, os.path.join(BASE_DIR, "exp1_speedup.png"))

    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
