#!/usr/bin/env python3
"""
scaling_parse.py — Parse scaling experiment logs into summary CSV.

Parses three types of lines from each log file:
  - GGML_PERF_ALL|nth=N|graph_us=...|avg_compute_us=...|avg_barrier_us=...|...
  - GGML_PERF_THREADS|nth=N|t0=...|t1=...|...        (per-thread compute_us)
  - GGML_PERF_BARRIER_THREADS|nth=N|t0=...|t1=...|... (per-thread barrier_us)
  - JSON lines: {"pp": ..., "tg": ..., "speed_pp": ..., "speed_tg": ..., ...}

Also supports legacy format (only GGML_PERF_ALL + GGML_PERF_THREADS).

Output:
  - scaling_summary.csv: per-config medians of throughput + perf breakdown
  - scaling_per_thread.csv: per-thread data for violin/box plots

Usage:
    python3 scaling_parse.py [--data-dir results] [--output-dir .]
    python3 scaling_parse.py --data-dir ../exp1_results/raw  # parse existing data
"""

import argparse
import json
import os
import sys
import csv
import re
from collections import defaultdict
from statistics import median, mean, stdev
import numpy as np


def parse_perf_kv(line):
    """Parse pipe-separated key=value line. Returns dict."""
    parts = line.strip().split("|")
    rec = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            try:
                rec[k] = float(v)
            except ValueError:
                rec[k] = v
    return rec


def parse_log(filepath):
    """
    Parse a single log file.
    Returns:
      - perf_all: list of dicts from GGML_PERF_ALL lines
      - perf_threads: list of dicts from GGML_PERF_THREADS lines
      - perf_barrier_threads: list of dicts from GGML_PERF_BARRIER_THREADS lines
      - jsonl: list of dicts from JSONL output
    """
    perf_all = []
    perf_threads = []
    perf_barrier_threads = []
    jsonl = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GGML_PERF_ALL|"):
                perf_all.append(parse_perf_kv(line))
            elif line.startswith("GGML_PERF_THREADS|"):
                perf_threads.append(parse_perf_kv(line))
            elif line.startswith("GGML_PERF_BARRIER_THREADS|"):
                perf_barrier_threads.append(parse_perf_kv(line))
            elif line.startswith("{"):
                try:
                    jsonl.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return perf_all, perf_threads, perf_barrier_threads, jsonl


def parse_filename(filename):
    """
    Parse filenames like:
      qwen4b_1n_fj_decode_r1.log       → (qwen4b, 1n, fj, decode, None, 1)
      qwen4b_8n_fj_96t_decode_r2.log   → (qwen4b, 8n, fj, decode, 96, 2)
      qwen3_4b_4n_fj_decode_r1.jsonl   → (qwen3_4b, 4n, fj, decode, None, 1)
      1numa_qwen4b_pl1.log             → (qwen4b, 1n, fj, decode, None, 1)  [exp_m1 format]
      4numa_llama8b_pl1.log            → (llama8b, 4n, fj, decode, None, 1)  [exp_m1 format]

    Returns: (model, numa, system, stage, threads_override, repeat) or None
    """
    base = os.path.splitext(filename)[0]

    # === Legacy exp_m1 format: <N>numa_<model>_pl<M>.log ===
    m_legacy = re.match(r'^(\d+)numa_(\w+)_pl(\d+)$', base)
    if m_legacy:
        n_numa = int(m_legacy.group(1))
        model = m_legacy.group(2)
        pl = int(m_legacy.group(3))
        numa = f"{n_numa}n"
        stage = "decode" if pl == 1 else "batch"
        return model, numa, "fj", stage, None, 1

    parts = base.split("_")

    # Find NUMA config
    numa_idx = None
    for i, p in enumerate(parts):
        if p in ("1n", "4n", "8n"):
            numa_idx = i
            break

    if numa_idx is None:
        return None

    model = "_".join(parts[:numa_idx])
    numa = parts[numa_idx]
    rest = parts[numa_idx + 1:]

    # rest could be: [fj, decode, r1] or [fj, 96t, decode, r1]
    if not rest:
        return None

    system = rest[0]

    # Check for thread override (e.g., "96t")
    threads_override = None
    offset = 1
    if len(rest) > 1 and re.match(r"^\d+t$", rest[1]):
        threads_override = int(rest[1][:-1])
        offset = 2

    if offset >= len(rest):
        return None

    stage = rest[offset]
    repeat = 1
    if offset + 1 < len(rest):
        r_str = rest[offset + 1]
        if r_str.startswith("r"):
            try:
                repeat = int(r_str[1:])
            except ValueError:
                pass

    return model, numa, system, stage, threads_override, repeat


def extract_tg_perf(perf_all_list, n_warmup=1):
    """
    Extract TG (decode) phase perf from GGML_PERF_ALL records.
    Assumes: first record(s) are PP (warmup/prefill), rest are TG.
    """
    if len(perf_all_list) <= n_warmup:
        return perf_all_list  # not enough, return all
    return perf_all_list[n_warmup:]


def extract_thread_matrix(perf_threads_list, n_warmup=1):
    """
    Build matrix (n_samples, nth) from GGML_PERF_THREADS records.
    Skip warmup records.
    """
    recs = perf_threads_list[n_warmup:]
    if not recs:
        return 0, np.array([])

    nth = int(recs[0].get("nth", 0))
    matrix = np.zeros((len(recs), nth))
    for i, rec in enumerate(recs):
        for t in range(nth):
            matrix[i, t] = rec.get(f"t{t}", 0)
    return nth, matrix


def compute_summary(perf_all_tg):
    """Compute summary stats from TG perf records."""
    if not perf_all_tg:
        return {}

    graph_us = [r.get("graph_us", 0) for r in perf_all_tg]
    compute_us = [r.get("avg_compute_us", 0) for r in perf_all_tg]
    barrier_us = [r.get("avg_barrier_us", 0) for r in perf_all_tg]
    idle_us = [r.get("avg_idle_us", 0) for r in perf_all_tg]
    max_compute = [r.get("max_compute_us", 0) for r in perf_all_tg]
    min_compute = [r.get("min_compute_us", 0) for r in perf_all_tg]

    return {
        "avg_graph_us": mean(graph_us),
        "avg_compute_us": mean(compute_us),
        "avg_barrier_us": mean(barrier_us),
        "avg_idle_us": mean(idle_us),
        "barrier_pct": mean(barrier_us) / mean(graph_us) * 100 if mean(graph_us) > 0 else 0,
        "compute_pct": mean(compute_us) / mean(graph_us) * 100 if mean(graph_us) > 0 else 0,
        "max_compute_us": mean(max_compute),
        "min_compute_us": mean(min_compute),
        "imbalance_ratio": mean(max_compute) / mean(min_compute) if mean(min_compute) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Parse scaling experiment logs")
    parser.add_argument("--data-dir", default="results", help="Directory containing log files")
    parser.add_argument("--output-dir", default=".", help="Directory for output CSVs")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_dir)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Collect all results
    all_configs = defaultdict(list)  # (model, numa, system, stage, threads) -> [run_data]

    for fname in sorted(os.listdir(data_dir)):
        if not (fname.endswith(".log") or fname.endswith(".jsonl")):
            continue

        parsed = parse_filename(fname)
        if parsed is None:
            print(f"  SKIP: {fname} (unparseable)", file=sys.stderr)
            continue

        model, numa, system, stage, threads_override, repeat = parsed
        filepath = os.path.join(data_dir, fname)

        perf_all, perf_threads, perf_barrier_threads, jsonl = parse_log(filepath)

        # Determine thread count
        nth = int(perf_all[0].get("nth", 0)) if perf_all else 0
        if threads_override:
            threads_key = threads_override
        else:
            threads_key = nth

        key = (model, numa, system, stage, threads_key)

        run_data = {
            "repeat": repeat,
            "nth": nth,
            "perf_all": perf_all,
            "perf_threads": perf_threads,
            "perf_barrier_threads": perf_barrier_threads,
            "jsonl": jsonl,
        }

        # Extract throughput from JSONL
        # llama-bench outputs separate rows for PP (n_gen=0) and TG (n_prompt=0)
        # llama-batched-bench outputs combined rows with speed_pp/speed_tg
        if jsonl:
            for j in jsonl:
                # llama-bench format: avg_ts is the throughput
                if j.get("n_gen", -1) == 0 and j.get("n_prompt", 0) > 0:
                    # This is a PP-only test
                    run_data["speed_pp"] = j.get("avg_ts", j.get("speed_pp", 0))
                elif j.get("n_prompt", -1) == 0 and j.get("n_gen", 0) > 0:
                    # This is a TG-only test
                    run_data["speed_tg"] = j.get("avg_ts", j.get("speed_tg", 0))
                else:
                    # llama-batched-bench format
                    run_data.setdefault("speed_pp", j.get("speed_pp", 0))
                    run_data.setdefault("speed_tg", j.get("speed_tg", 0))
                    run_data.setdefault("speed_total", j.get("speed", 0))

        all_configs[key].append(run_data)

    print(f"Parsed {sum(len(v) for v in all_configs.values())} runs across {len(all_configs)} configs")

    # ======================================================================
    # Output 1: scaling_summary.csv
    # ======================================================================
    summary_rows = []
    for key, runs in sorted(all_configs.items()):
        model, numa, system, stage, threads = key

        # Aggregate perf across runs
        all_perf_tg = []
        speeds_tg = []
        speeds_pp = []
        for run in runs:
            tg_perf = extract_tg_perf(run["perf_all"], n_warmup=1)
            all_perf_tg.extend(tg_perf)
            if "speed_tg" in run:
                speeds_tg.append(run["speed_tg"])
            if "speed_pp" in run:
                speeds_pp.append(run["speed_pp"])

        perf = compute_summary(all_perf_tg)

        row = {
            "model": model,
            "numa": numa,
            "system": system,
            "stage": stage,
            "threads": threads,
            "n_runs": len(runs),
            "speed_tg": round(median(speeds_tg), 3) if speeds_tg else 0,
            "speed_pp": round(median(speeds_pp), 2) if speeds_pp else 0,
        }
        row.update({k: round(v, 2) for k, v in perf.items()})
        summary_rows.append(row)

    summary_csv = os.path.join(out_dir, "scaling_summary.csv")
    if summary_rows:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Written: {summary_csv} ({len(summary_rows)} rows)")

    # ======================================================================
    # Output 2: scaling_per_thread.csv  (for violin/box plots)
    # ======================================================================
    thread_rows = []
    for key, runs in sorted(all_configs.items()):
        model, numa, system, stage, threads = key

        for run in runs:
            # Get TG-phase per-thread data
            nth, compute_matrix = extract_thread_matrix(run["perf_threads"], n_warmup=1)
            _, barrier_matrix = extract_thread_matrix(run.get("perf_barrier_threads", []), n_warmup=1)

            if compute_matrix.size == 0:
                continue

            # Average across graph calls for each thread
            thread_means = np.mean(compute_matrix, axis=0)  # (nth,)
            barrier_means = np.mean(barrier_matrix, axis=0) if barrier_matrix.size > 0 else np.zeros(nth)

            for t in range(nth):
                thread_rows.append({
                    "model": model,
                    "numa": numa,
                    "system": system,
                    "stage": stage,
                    "threads": threads,
                    "repeat": run["repeat"],
                    "thread_id": t,
                    "numa_domain": t // 24,  # 24 cores per socket
                    "compute_us": round(thread_means[t], 1),
                    "barrier_us": round(barrier_means[t], 1) if t < len(barrier_means) else 0,
                })

    thread_csv = os.path.join(out_dir, "scaling_per_thread.csv")
    if thread_rows:
        with open(thread_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(thread_rows[0].keys()))
            writer.writeheader()
            writer.writerows(thread_rows)
        print(f"Written: {thread_csv} ({len(thread_rows)} rows)")

    # ======================================================================
    # Print summary table
    # ======================================================================
    print("\n" + "=" * 110)
    print("SCALING SUMMARY — Fork-Join Pure (non-TP)")
    print("=" * 110)
    print(f"{'Model':<12} {'NUMA':<6} {'Sys':<6} {'Stage':<8} {'Thds':<6} "
          f"{'TG t/s':<8} {'PP t/s':<8} {'Barr%':<7} {'Comp%':<7} {'Imbal':<7} "
          f"{'Graph_us':<10} {'N':<3}")
    print("-" * 110)

    for r in summary_rows:
        print(f"{r['model']:<12} {r['numa']:<6} {r['system']:<6} {r['stage']:<8} "
              f"{r['threads']:<6} {r['speed_tg']:<8} {r['speed_pp']:<8} "
              f"{r.get('barrier_pct', 0):<7.1f} {r.get('compute_pct', 0):<7.1f} "
              f"{r.get('imbalance_ratio', 0):<7.2f} {r.get('avg_graph_us', 0):<10.0f} "
              f"{r['n_runs']:<3}")


if __name__ == "__main__":
    main()
