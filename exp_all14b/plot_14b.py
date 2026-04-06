#!/usr/bin/env python3
"""
exp1_plot_14b.py — Generate figures for 14B row-shard-cache experiment.

Generates:
  Figure 1: Decode throughput (PP + TG) — grouped bar chart across NUMA configs
  Figure 2: Batched decode throughput — line chart across batch sizes
  Figure 3: Row-shard-cache speedup heatmap
"""

import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = BASE_DIR  # JSONL files are directly in this directory


# ── Helper: parse JSONL files ──
def load_jsonl(path):
    """Load all JSON lines from file, return list of dicts."""
    results = []
    try:
        with open(path, "rb") as f:
            for line in f:
                line = line.decode("utf-8", errors="ignore").strip()
                if line.startswith("{"):
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return results


def get_last_by_pl(results):
    """From a list of JSON results, take the last occurrence per pl value."""
    by_pl = {}
    for r in results:
        by_pl[r["pl"]] = r
    return by_pl


# ── Load all data ──
def load_all_data():
    """Load and organize all 14B experiment data."""
    data = {}

    # 1NUMA (from repeat 1, these used PP=64 TG=64 for decode, PP=128/256/512 TG=1 for prefill)
    # We only use the decode test for 1N single-request PP/TG
    for sys_key, prefix in [("fj", "qwen25_14b_1n_fj"), ("task", "qwen25_14b_1n_task")]:
        # Decode: take r1
        results = load_jsonl(os.path.join(RAW_DIR, f"{prefix}_decode_r1.jsonl"))
        if results:
            r = results[-1]  # last result
            data[("1n", sys_key, "decode")] = {"pp": r["speed_pp"], "tg": r["speed_tg"]}

        # Batch: take r1
        results = load_jsonl(os.path.join(RAW_DIR, f"{prefix}_batch_r1.jsonl"))
        by_pl = get_last_by_pl(results)
        for pl, r in by_pl.items():
            data[("1n", sys_key, "batch", pl)] = {"pp": r["speed_pp"], "tg": r["speed_tg"]}

    # 4NUMA decode (from v1 run, PP=64 TG=8)
    decode_files_4n = {
        "fj":      "qwen25_14b_4n_fj_decode_r1.jsonl",
        "fjtp":    "qwen25_14b_4n_fjtp_decode_r1.jsonl",
        "fjtp_rs": "qwen25_14b_4n_fjtp_rs_decode_r1.jsonl",
        "task_rs": "qwen25_14b_4n_task_rs_decode_r1.jsonl",
    }
    for sys_key, fname in decode_files_4n.items():
        results = load_jsonl(os.path.join(RAW_DIR, fname))
        if results:
            r = results[-1]
            data[("4n", sys_key, "decode")] = {"pp": r["speed_pp"], "tg": r["speed_tg"]}

    # 4NUMA prefill v2 (PP=64 TG=8)
    prefill_files_4n = {
        "fj":      "qwen25_14b_4n_fj_prefill_v2_r1.jsonl",
        "fjtp_rs": "qwen25_14b_4n_fjtp_rs_prefill_v2_r1.jsonl",
        "task_rs": "qwen25_14b_4n_task_rs_prefill_v2_r1.jsonl",
    }
    for sys_key, fname in prefill_files_4n.items():
        results = load_jsonl(os.path.join(RAW_DIR, fname))
        if results:
            r = results[-1]
            data[("4n", sys_key, "prefill")] = {"pp": r["speed_pp"], "tg": r["speed_tg"]}

    # 4NUMA batch
    batch_files_4n = {
        "fjtp_rs": "qwen25_14b_4n_fjtp_rs_batch_r1.jsonl",
        "task_rs": "qwen25_14b_4n_task_rs_batch_r1.jsonl",
    }
    for sys_key, fname in batch_files_4n.items():
        results = load_jsonl(os.path.join(RAW_DIR, fname))
        by_pl = get_last_by_pl(results)
        for pl, r in by_pl.items():
            data[("4n", sys_key, "batch", pl)] = {"pp": r["speed_pp"], "tg": r["speed_tg"]}

    # 8NUMA decode
    decode_files_8n = {
        "fj":      "qwen25_14b_8n_fj_decode_r1.jsonl",
        "fjtp":    "qwen25_14b_8n_fjtp_decode_r1.jsonl",
        "fjtp_rs": "qwen25_14b_8n_fjtp_rs_decode_r1.jsonl",
        "task_rs": "qwen25_14b_8n_task_rs_decode_r1.jsonl",
    }
    for sys_key, fname in decode_files_8n.items():
        results = load_jsonl(os.path.join(RAW_DIR, fname))
        if results:
            r = results[-1]
            data[("8n", sys_key, "decode")] = {"pp": r["speed_pp"], "tg": r["speed_tg"]}

    # 8NUMA prefill v2
    prefill_files_8n = {
        "fj":      "qwen25_14b_8n_fj_prefill_r1.jsonl",
        "fjtp_rs": "qwen25_14b_8n_fjtp_rs_prefill_r1.jsonl",
        "task_rs": "qwen25_14b_8n_task_rs_prefill_r1.jsonl",
    }
    for sys_key, fname in prefill_files_8n.items():
        results = load_jsonl(os.path.join(RAW_DIR, fname))
        if results:
            r = results[-1]
            data[("8n", sys_key, "prefill")] = {"pp": r["speed_pp"], "tg": r["speed_tg"]}

    # 8NUMA batch
    batch_files_8n = {
        "fjtp_rs": "qwen25_14b_8n_fjtp_rs_batch_r1.jsonl",
        "task_rs": "qwen25_14b_8n_task_rs_batch_r1.jsonl",
    }
    for sys_key, fname in batch_files_8n.items():
        results = load_jsonl(os.path.join(RAW_DIR, fname))
        by_pl = get_last_by_pl(results)
        for pl, r in by_pl.items():
            data[("8n", sys_key, "batch", pl)] = {"pp": r["speed_pp"], "tg": r["speed_tg"]}

    return data


# ── Styles ──
SYSTEM_COLORS = {
    "fj":      "#d62728",   # red
    "fjtp":    "#ff7f0e",   # orange
    "fjtp_rs": "#1f77b4",   # blue
    "task_rs": "#2ca02c",   # green
    "task":    "#9467bd",   # purple
}
SYSTEM_LABELS = {
    "fj":      "FJ-Pure (no TP)",
    "fjtp":    "FJ + TP (no RS)",
    "fjtp_rs": "FJ + TP + RS",
    "task_rs": "TaskInfer + RS",
    "task":    "TaskInfer (1N)",
}
NUMA_LABELS = {
    "1n": "1 NUMA\n(24 threads)",
    "4n": "4 NUMA\n(96 threads)",
    "8n": "8 NUMA\n(192 threads)",
}


def plot_fig1_decode_tg(data, output_path):
    """
    Figure 1: Decode TG throughput across NUMA configs.
    Shows single-request token generation speed.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1NUMA
    ax = axes[0]
    systems_1n = ["fj", "task"]
    labels_1n = ["FJ-Pure", "TaskInfer"]
    vals = [data.get(("1n", s, "decode"), {}).get("tg", 0) for s in systems_1n]
    colors = [SYSTEM_COLORS[s] for s in systems_1n]
    bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.85, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(labels_1n)))
    ax.set_xticklabels(labels_1n, fontsize=9)
    ax.set_title("1 NUMA (24 threads)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Decode Throughput (tokens/s)", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # 4NUMA and 8NUMA
    for idx, numa in enumerate(["4n", "8n"]):
        ax = axes[idx + 1]
        tp_n = "4" if numa == "4n" else "8"
        systems = ["fj", "fjtp", "fjtp_rs", "task_rs"]
        labels = ["FJ-Pure\n(no TP)", f"FJ+TP{tp_n}\n(no RS)", f"FJ+TP{tp_n}\n+RS", f"TaskInfer\n+RS"]
        vals = [data.get((numa, s, "decode"), {}).get("tg", 0) for s in systems]
        colors = [SYSTEM_COLORS[s] for s in systems]
        bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.85, width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        n_threads = "96" if numa == "4n" else "192"
        ax.set_title(f"{numa[0]} NUMA ({n_threads} threads)", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Qwen2.5-14B (F16, 29.5GB) — Single-Request Decode Throughput\n"
                 "PP=64, TG=8 · Intel Xeon Platinum 8160 · --numa distribute",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_fig2_prefill_pp(data, output_path):
    """
    Figure 2: Prefill PP throughput across NUMA configs.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1NUMA - use decode test PP speed (PP=64)
    ax = axes[0]
    systems_1n = ["fj", "task"]
    labels_1n = ["FJ-Pure", "TaskInfer"]
    vals = [data.get(("1n", s, "decode"), {}).get("pp", 0) for s in systems_1n]
    colors = [SYSTEM_COLORS[s] for s in systems_1n]
    bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.85, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(labels_1n)))
    ax.set_xticklabels(labels_1n, fontsize=9)
    ax.set_title("1 NUMA (24 threads)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Prefill Throughput (tokens/s)", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # 4NUMA and 8NUMA
    for idx, numa in enumerate(["4n", "8n"]):
        ax = axes[idx + 1]
        tp_n = "4" if numa == "4n" else "8"
        systems = ["fj", "fjtp_rs", "task_rs"]
        labels = ["FJ-Pure\n(no TP)", f"FJ+TP{tp_n}\n+RS", f"TaskInfer\n+RS"]
        vals = [data.get((numa, s, "decode"), {}).get("pp", 0) for s in systems]
        colors = [SYSTEM_COLORS[s] for s in systems]
        bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.85, width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        n_threads = "96" if numa == "4n" else "192"
        ax.set_title(f"{numa[0]} NUMA ({n_threads} threads)", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Qwen2.5-14B (F16, 29.5GB) — Prefill Throughput\n"
                 "PP=64, TG=8 · Intel Xeon Platinum 8160 · --numa distribute",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_fig3_batch(data, output_path):
    """
    Figure 3: Batched decode throughput — TG t/s vs batch size.
    Separate panels for 4NUMA and 8NUMA.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    batches = [1, 4, 8, 16]

    for idx, numa in enumerate(["4n", "8n"]):
        ax = axes[idx]
        tp_n = "4" if numa == "4n" else "8"

        for sys_key, label, color, marker in [
            ("fjtp_rs", f"FJ+TP{tp_n}+RS", SYSTEM_COLORS["fjtp_rs"], "s"),
            ("task_rs", f"TaskInfer+RS", SYSTEM_COLORS["task_rs"], "o"),
        ]:
            vals = []
            for pl in batches:
                v = data.get((numa, sys_key, "batch", pl), {}).get("tg", 0)
                vals.append(v)

            ax.plot(batches, vals, f"{marker}-", color=color,
                    linewidth=2.5, markersize=8, label=label)

            for b, v in zip(batches, vals):
                if v > 0:
                    ax.annotate(f"{v:.2f}", (b, v), textcoords="offset points",
                               xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

        # Add ideal linear scaling reference from pl=1 (FJ+RS)
        base_tg = data.get((numa, "fjtp_rs", "batch", 1), {}).get("tg", 0)
        if base_tg > 0:
            ideal = [base_tg * pl for pl in batches]
            ax.plot(batches, ideal, "--", color="gray", linewidth=1, alpha=0.5, label="Ideal linear (FJ+RS)")

        n_threads = "96" if numa == "4n" else "192"
        ax.set_title(f"{numa[0]} NUMA ({n_threads} threads, TP{tp_n})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Batch Size (parallel users)", fontsize=11)
        ax.set_ylabel("TG Throughput (tokens/s)" if idx == 0 else "", fontsize=11)
        ax.set_xticks(batches)
        ax.grid(linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc="upper left")

    fig.suptitle("Qwen2.5-14B (F16) — Batched Decode Throughput\n"
                 "PP=64, TG=32 · row-shard-cache · Intel Xeon Platinum 8160",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_fig4_rs_speedup(data, output_path):
    """
    Figure 4: row-shard-cache speedup — comparing with/without RS.
    Bar chart showing speedup factors.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Decode TG speedup
    ax = axes[0]
    configs = []
    speedups_rs = []
    speedups_tp_only = []
    labels = []

    for numa, tp_n in [("4n", "4"), ("8n", "8")]:
        fj_tg = data.get((numa, "fj", "decode"), {}).get("tg", 0)
        fjtp_tg = data.get((numa, "fjtp", "decode"), {}).get("tg", 0)
        fjtp_rs_tg = data.get((numa, "fjtp_rs", "decode"), {}).get("tg", 0)
        task_rs_tg = data.get((numa, "task_rs", "decode"), {}).get("tg", 0)

        n_t = "96" if numa == "4n" else "192"
        labels.append(f"{numa[0]}N FJ+TP{tp_n}+RS\nvs FJ-Pure")
        speedups_rs.append(fjtp_rs_tg / fj_tg if fj_tg > 0 else 0)

        labels.append(f"{numa[0]}N TaskInfer+RS\nvs FJ-Pure")
        speedups_rs.append(task_rs_tg / fj_tg if fj_tg > 0 else 0)

    x = np.arange(len(labels))
    colors_list = [SYSTEM_COLORS["fjtp_rs"], SYSTEM_COLORS["task_rs"]] * 2
    bars = ax.bar(x, speedups_rs, color=colors_list, alpha=0.85, width=0.6)
    for bar, v in zip(bars, speedups_rs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{v:.1f}x", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Speedup (x)", fontsize=11)
    ax.set_title("Decode (TG) Speedup vs FJ-Pure", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Prefill PP speedup
    ax = axes[1]
    labels2 = []
    speedups2 = []

    for numa, tp_n in [("4n", "4"), ("8n", "8")]:
        fj_pp = data.get((numa, "fj", "decode"), {}).get("pp", 0)
        fjtp_rs_pp = data.get((numa, "fjtp_rs", "decode"), {}).get("pp", 0)
        task_rs_pp = data.get((numa, "task_rs", "decode"), {}).get("pp", 0)

        labels2.append(f"{numa[0]}N FJ+TP{tp_n}+RS\nvs FJ-Pure")
        speedups2.append(fjtp_rs_pp / fj_pp if fj_pp > 0 else 0)

        labels2.append(f"{numa[0]}N TaskInfer+RS\nvs FJ-Pure")
        speedups2.append(task_rs_pp / fj_pp if fj_pp > 0 else 0)

    x2 = np.arange(len(labels2))
    bars = ax.bar(x2, speedups2, color=colors_list, alpha=0.85, width=0.6)
    for bar, v in zip(bars, speedups2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{v:.1f}x", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(x2)
    ax.set_xticklabels(labels2, fontsize=8)
    ax.set_ylabel("Speedup (x)", fontsize=11)
    ax.set_title("Prefill (PP) Speedup vs FJ-Pure", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.suptitle("Qwen2.5-14B — row-shard-cache Speedup over Baseline\n"
                 "Intel Xeon Platinum 8160 · >1.0x = faster than FJ-Pure",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_fig5_all_models_decode(data, output_path):
    """
    Figure 5: Cross-model comparison of decode TG on multi-NUMA.
    Include 4B/8B data from existing CSV if available.
    """
    csv_path = os.path.join(BASE_DIR, "exp1_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping Figure 5: {csv_path} not found (run exp1_parse.py first)")
        return

    import csv as csv_mod
    with open(csv_path) as f:
        csv_data = list(csv_mod.DictReader(f))

    def get_csv_val(model, numa, system, stage, metric):
        for row in csv_data:
            if (row["model"] == model and row["numa"] == numa and
                row["system"] == system and row["stage"] == stage and
                int(row["batch"]) == 1):
                return float(row[metric])
        return 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 4NUMA
    for idx, (numa, tp_n) in enumerate([("4n", "4"), ("8n", "8")]):
        ax = axes[idx]
        models = ["Qwen3-4B", "LLaMA3-8B", "Qwen2.5-14B"]
        model_keys = ["qwen3_4b", "llama3_8b", "qwen25_14b"]

        fj_vals = []
        fjtp_vals = []
        task_vals = []

        for mk in model_keys:
            if mk == "qwen25_14b":
                # Use our 14B data with RS
                fj_vals.append(data.get((numa, "fj", "decode"), {}).get("tg", 0))
                fjtp_vals.append(data.get((numa, "fjtp_rs", "decode"), {}).get("tg", 0))
                task_vals.append(data.get((numa, "task_rs", "decode"), {}).get("tg", 0))
            else:
                fj_vals.append(get_csv_val(mk, numa, "fj", "decode", "speed_tg_median"))
                fjtp_vals.append(get_csv_val(mk, numa, "fjtp", "decode", "speed_tg_median"))
                task_vals.append(get_csv_val(mk, numa, "task", "decode", "speed_tg_median"))

        x = np.arange(len(models))
        width = 0.25
        bars1 = ax.bar(x - width, fj_vals, width, color=SYSTEM_COLORS["fj"],
                       alpha=0.85, label="FJ-Pure")
        bars2 = ax.bar(x, fjtp_vals, width, color=SYSTEM_COLORS["fjtp_rs"],
                       alpha=0.85, label=f"FJ+TP{tp_n} (+RS for 14B)")
        bars3 = ax.bar(x + width, task_vals, width, color=SYSTEM_COLORS["task_rs"],
                       alpha=0.85, label=f"TaskInfer (+RS for 14B)")

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                            f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        n_t = "96" if numa == "4n" else "192"
        ax.set_title(f"{numa[0]} NUMA ({n_t} threads)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel("Decode TG (tokens/s)" if idx == 0 else "", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Cross-Model Decode Throughput on Multi-NUMA\n"
                 "14B uses row-shard-cache; 4B/8B use standard TP · Intel Xeon Platinum 8160",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    data = load_all_data()
    print(f"Loaded {len(data)} data points")

    # Print summary table
    print("\n" + "=" * 80)
    print("14B RESULTS SUMMARY")
    print("=" * 80)

    print("\n--- Single Request (PP=64, TG=8/64) ---")
    print(f"{'NUMA':<6} {'System':<18} {'PP t/s':<10} {'TG t/s':<10}")
    print("-" * 50)
    for numa in ["1n", "4n", "8n"]:
        systems = ["fj", "task"] if numa == "1n" else ["fj", "fjtp", "fjtp_rs", "task_rs"]
        for s in systems:
            d = data.get((numa, s, "decode"), {})
            if d:
                label = SYSTEM_LABELS.get(s, s)
                print(f"{numa:<6} {label:<18} {d['pp']:<10.1f} {d['tg']:<10.2f}")
        print()

    print("\n--- Batch (PP=64, TG=32) ---")
    print(f"{'NUMA':<6} {'System':<18} {'pl=1':<8} {'pl=4':<8} {'pl=8':<8} {'pl=16':<8}")
    print("-" * 60)
    for numa in ["4n", "8n"]:
        for s in ["fjtp_rs", "task_rs"]:
            label = SYSTEM_LABELS.get(s, s)
            vals = []
            for pl in [1, 4, 8, 16]:
                v = data.get((numa, s, "batch", pl), {}).get("tg", 0)
                vals.append(f"{v:.2f}" if v > 0 else "-")
            print(f"{numa:<6} {label:<18} {'  '.join(f'{v:<6}' for v in vals)}")
        print()

    # Generate figures
    plot_fig1_decode_tg(data, os.path.join(BASE_DIR, "fig_14b_decode_tg.png"))
    plot_fig2_prefill_pp(data, os.path.join(BASE_DIR, "fig_14b_prefill_pp.png"))
    plot_fig3_batch(data, os.path.join(BASE_DIR, "fig_14b_batch.png"))
    plot_fig4_rs_speedup(data, os.path.join(BASE_DIR, "fig_14b_rs_speedup.png"))

    print("\nAll 14B figures generated!")


if __name__ == "__main__":
    main()
