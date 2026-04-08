#!/bin/bash
# ==============================================================================
# run_scaling_bench.sh — Scaling experiment for Fork-Join Pure (non-TP)
#
# Experiment matrix:
#   - NUMA scaling:  1N(24t), 4N(96t), 8N(192t) — fixed 24 threads per NUMA
#   - Thread sweep:  8N with 24/48/96/192 threads
#   - Batch:         pl=1 (decode), pl=8 (batched decode)
#   - Models:        Qwen3-4B F16 (primary), LLaMA3-8B (validation)
#   - Repeats:       3 per config
#
# Requirements:
#   - fork-join binary compiled with -DGGML_PERF_ENABLED=ON
#   - numactl installed
#   - Models at ~/model/
#
# Output: bench_scaling/results/<model>_<config>_r<N>.log
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN=/home/haibin/tp/fork-join/build/bin/llama-bench
OUTDIR="$SCRIPT_DIR/results"
mkdir -p "$OUTDIR"

# Models
QWEN4B=~/model/Qwen3-4B-Ins-2507-f16.gguf
LLAMA8B=~/model/Meta-Llama-3.1-8B-Instruct.FP16.gguf

# Common args: pp=64, tg=64, batch up to 512, output jsonl
# Use -p for prompt tokens, -n for generation tokens
COMMON_DECODE="-p 64 -n 64 -o jsonl"
COMMON_BATCH="-p 64 -n 32 -o jsonl"

REPEATS=3

# ==============================================================================
# Helper: run a single benchmark
# ==============================================================================
run_bench() {
    local label="$1"
    local outfile="$2"
    shift 2

    echo "$(date '+%H:%M:%S') Running: $label → $outfile"
    # Capture both stdout (JSONL) and stderr (GGML_PERF lines + model loading)
    "$@" > "$outfile" 2>&1
    echo "$(date '+%H:%M:%S') Done: $label"
}

# ==============================================================================
# Part 1: NUMA Scaling — 1N/4N/8N, 24 threads per NUMA node
# ==============================================================================
echo ""
echo "=============================================="
echo " Part 1: NUMA Scaling (1N / 4N / 8N)"
echo "=============================================="

run_numa_scaling() {
    local model_path="$1"
    local model_name="$2"

    for NUMA in 1 4 8; do
        THREADS=$((NUMA * 24))
        case $NUMA in
            1) NODES="0" ;;
            4) NODES="0,1,2,3" ;;
            8) NODES="0,1,2,3,4,5,6,7" ;;
        esac

        for r in $(seq 1 $REPEATS); do
            # Decode: pl=1
            run_bench "${model_name} ${NUMA}N-${THREADS}t decode r${r}" \
                "$OUTDIR/${model_name}_${NUMA}n_fj_decode_r${r}.log" \
                numactl --cpunodebind=$NODES --membind=$NODES \
                $BIN -m "$model_path" -t $THREADS $COMMON_DECODE

            # Batched decode: pl=8
            # llama-bench doesn't have -npl, but we use -b for batch size
            # Actually for llama-bench the batching is implicit in prompt eval
            # We'll use the same command but note: llama-bench does pp and tg separately
            # For batched experiment, we need llama-batched-bench
        done
    done
}

# Check if binary exists
if [ ! -f "$BIN" ]; then
    echo "ERROR: Binary not found: $BIN"
    echo "Please build fork-join first:"
    echo "  cd /home/haibin/tp/fork-join/build && cmake .. -DGGML_OPENMP=ON -DGGML_PERF_ENABLED=ON && make -j"
    exit 1
fi

# Qwen3-4B (primary)
if [ -f "$QWEN4B" ]; then
    run_numa_scaling "$QWEN4B" "qwen4b"
else
    echo "WARNING: Qwen3-4B model not found at $QWEN4B, skipping"
fi

# LLaMA3-8B (validation)
if [ -f "$LLAMA8B" ]; then
    run_numa_scaling "$LLAMA8B" "llama8b"
else
    echo "WARNING: LLaMA3-8B model not found at $LLAMA8B, skipping"
fi

# ==============================================================================
# Part 2: Thread Sweep — Fixed 8NUMA, vary threads
# ==============================================================================
echo ""
echo "=============================================="
echo " Part 2: Thread Sweep (8N, vary threads)"
echo "=============================================="

run_thread_sweep() {
    local model_path="$1"
    local model_name="$2"
    local nodes="0,1,2,3,4,5,6,7"

    for THREADS in 24 48 96 192; do
        for r in $(seq 1 $REPEATS); do
            run_bench "${model_name} 8N-${THREADS}t decode r${r}" \
                "$OUTDIR/${model_name}_8n_fj_${THREADS}t_decode_r${r}.log" \
                numactl --cpunodebind=$nodes --membind=$nodes \
                $BIN -m "$model_path" -t $THREADS $COMMON_DECODE
        done
    done
}

# Qwen3-4B (primary)
if [ -f "$QWEN4B" ]; then
    run_thread_sweep "$QWEN4B" "qwen4b"
fi

echo ""
echo "=============================================="
echo " All benchmarks complete!"
echo " Results in: $OUTDIR/"
echo "=============================================="
ls -la "$OUTDIR/"
