#!/bin/bash
# run_bench.sh — Run compute/barrier/idle breakdown benchmark
# 3 models × PP64 TG32 × Batch 1,2,4,8,16,32
# 1NUMA (node 1), 24 threads (physical cores)

set -euo pipefail

BIN=/home/haibin/tp/fork-join/build/bin/llama-batched-bench
OUTDIR="$(dirname "$0")/raw_data"
mkdir -p "$OUTDIR"

NUMA="numactl --cpunodebind=1 --membind=1"
THREADS=24
COMMON="-c 4096 -b 512 -ub 256 -npp 64 -ntg 32 -npl 1,2,4,8,16,32 --output-format jsonl"

declare -A MODELS
MODELS[qwen3_06b]="$HOME/model/Qwen3-Embedding-0.6B-f16.gguf"
MODELS[qwen3_4b]="$HOME/model/Qwen3-4B-Ins-2507-f16.gguf"
MODELS[llama3_8b]="$HOME/model/Meta-Llama-3.1-8B-Instruct.FP16.gguf"

for name in qwen3_06b qwen3_4b llama3_8b; do
    model_path="${MODELS[$name]}"
    outfile="$OUTDIR/${name}.log"

    echo "========================================"
    echo "Running: $name"
    echo "  Model: $model_path"
    echo "  Output: $outfile"
    echo "========================================"

    # Run benchmark — capture stdout+stderr together so we get both JSONL and GGML_PERF lines
    $NUMA $BIN \
        -m "$model_path" \
        -t $THREADS \
        $COMMON \
        > "$outfile" 2>&1

    echo "Done: $name"
    echo ""
done

echo "All benchmarks complete. Raw data in: $OUTDIR/"
ls -la "$OUTDIR/"
