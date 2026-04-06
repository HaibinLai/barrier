#!/bin/bash
# =================================================================
# exp1_run_full.sh — Experiment 1: Overall Performance
# 3 Systems × 2 Models × 3 NUMA configs
# Uses: fork-join binary (FJ Pure, FJ+TP) + tp-llama binary (TaskInfer)
# =================================================================
set -euo pipefail

# ======================== Configuration ==========================
FJ_BIN=/home/haibin/tp/fork-join/build/bin/llama-batched-bench
TF_BIN=/home/haibin/tp/tp-llama/build/bin/llama-batched-bench

OUTDIR=/home/haibin/tp/exp1_results/raw
LOGFILE=/home/haibin/tp/exp1_results/exp1.log
mkdir -p "$OUTDIR"

# Models
declare -A MODELS
MODELS[qwen3_4b]="$HOME/model/Qwen3-4B-Ins-2507-f16.gguf"
MODELS[llama3_8b]="$HOME/model/Meta-Llama-3.1-8B-Instruct.FP16.gguf"

MODEL_ORDER=(qwen3_4b llama3_8b)

REPEATS=3
COMMON="-c 4096 -b 512 -ub 256 --output-format jsonl"

# ======================== Functions ==============================

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGFILE"
}

run_bench() {
    local label="$1"
    local outfile="$2"
    shift 2

    log "  RUN: $label -> $(basename $outfile)"

    # Run benchmark, capture both stdout and stderr
    if "$@" > "$outfile" 2>&1; then
        # Extract speed from JSONL output
        local speeds=$(grep "^{" "$outfile" | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line.strip())
    print(f'  B={d[\"pl\"]}: PP={d[\"speed_pp\"]:.1f} TG={d[\"speed_tg\"]:.2f} t/s')
" 2>/dev/null)
        log "$speeds"
    else
        log "  FAILED: exit $?"
    fi
}

# ======================== Main Loop ==============================

log "=========================================="
log "Experiment 1: Overall Performance"
log "Start: $(date)"
log "Repeats: $REPEATS"
log "=========================================="

for repeat in $(seq 1 $REPEATS); do
    log ""
    log "############## REPEAT $repeat/$REPEATS ##############"

    # ============================================================
    # 1 NUMA (node 0), 24 threads, TP=1
    # ============================================================
    NUMA="numactl --cpunodebind=0 --membind=0"
    THREADS=24

    log ""
    log "====== 1NUMA, 24 threads ======"

    for model_name in "${MODEL_ORDER[@]}"; do
        model_path="${MODELS[$model_name]}"
        log ""
        log "  Model: $model_name"

        # --- Decode ---
        # Group A: FJ Pure (no TP, 1NUMA)
        run_bench "1N FJ-Pure decode" \
            "$OUTDIR/${model_name}_1n_fj_decode_r${repeat}.jsonl" \
            $NUMA $FJ_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 64 -npl 1 $COMMON --numa distribute

        # Group C: TaskInfer (no TP on 1NUMA, just taskflow exec model)
        run_bench "1N TaskInfer decode" \
            "$OUTDIR/${model_name}_1n_task_decode_r${repeat}.jsonl" \
            $NUMA $TF_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 64 -npl 1 $COMMON --cpu-strategy taskflow --numa distribute

        # --- Prefill ---
        run_bench "1N FJ-Pure prefill" \
            "$OUTDIR/${model_name}_1n_fj_prefill_r${repeat}.jsonl" \
            $NUMA $FJ_BIN -m "$model_path" -t $THREADS \
            -npp 128,256,512 -ntg 1 -npl 1 $COMMON --numa distribute

        run_bench "1N TaskInfer prefill" \
            "$OUTDIR/${model_name}_1n_task_prefill_r${repeat}.jsonl" \
            $NUMA $TF_BIN -m "$model_path" -t $THREADS \
            -npp 128,256,512 -ntg 1 -npl 1 $COMMON --cpu-strategy taskflow --numa distribute

        # --- Batched Decode ---
        run_bench "1N FJ-Pure batch" \
            "$OUTDIR/${model_name}_1n_fj_batch_r${repeat}.jsonl" \
            $NUMA $FJ_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON --numa distribute

        run_bench "1N TaskInfer batch" \
            "$OUTDIR/${model_name}_1n_task_batch_r${repeat}.jsonl" \
            $NUMA $TF_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON --cpu-strategy taskflow --numa distribute
    done

    # ============================================================
    # 4 NUMA (nodes 0-3), 96 threads, TP=4
    # ============================================================
    NUMA="numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3"
    THREADS=96
    TP=4

    log ""
    log "====== 4NUMA, 96 threads, TP${TP} ======"

    for model_name in "${MODEL_ORDER[@]}"; do
        model_path="${MODELS[$model_name]}"
        log ""
        log "  Model: $model_name"

        # --- Decode ---
        # Group A: FJ Pure (no TP)
        run_bench "4N FJ-Pure decode" \
            "$OUTDIR/${model_name}_4n_fj_decode_r${repeat}.jsonl" \
            $NUMA $FJ_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 64 -npl 1 $COMMON --numa distribute

        # Group B: FJ+TP
        run_bench "4N FJ+TP${TP} decode" \
            "$OUTDIR/${model_name}_4n_fjtp_decode_r${repeat}.jsonl" \
            $NUMA $FJ_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 64 -npl 1 $COMMON \
            --cpu-tp $TP --cpu-tp-runtime forkjoin --numa distribute

        # Group C: TaskInfer
        run_bench "4N TaskInfer decode" \
            "$OUTDIR/${model_name}_4n_task_decode_r${repeat}.jsonl" \
            $NUMA $TF_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 64 -npl 1 $COMMON \
            --cpu-tp $TP --cpu-strategy taskflow --numa distribute

        # --- Prefill ---
        run_bench "4N FJ-Pure prefill" \
            "$OUTDIR/${model_name}_4n_fj_prefill_r${repeat}.jsonl" \
            $NUMA $FJ_BIN -m "$model_path" -t $THREADS \
            -npp 128,256,512 -ntg 1 -npl 1 $COMMON --numa distribute

        run_bench "4N FJ+TP${TP} prefill" \
            "$OUTDIR/${model_name}_4n_fjtp_prefill_r${repeat}.jsonl" \
            $NUMA $FJ_BIN -m "$model_path" -t $THREADS \
            -npp 128,256,512 -ntg 1 -npl 1 $COMMON \
            --cpu-tp $TP --cpu-tp-runtime forkjoin --numa distribute

        run_bench "4N TaskInfer prefill" \
            "$OUTDIR/${model_name}_4n_task_prefill_r${repeat}.jsonl" \
            $NUMA $TF_BIN -m "$model_path" -t $THREADS \
            -npp 128,256,512 -ntg 1 -npl 1 $COMMON \
            --cpu-tp $TP --cpu-strategy taskflow --numa distribute

        # --- Batched Decode ---
        run_bench "4N FJ-Pure batch" \
            "$OUTDIR/${model_name}_4n_fj_batch_r${repeat}.jsonl" \
            $NUMA $FJ_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON --numa distribute

        run_bench "4N FJ+TP${TP} batch" \
            "$OUTDIR/${model_name}_4n_fjtp_batch_r${repeat}.jsonl" \
            $NUMA $FJ_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON \
            --cpu-tp $TP --cpu-tp-runtime forkjoin --numa distribute

        run_bench "4N TaskInfer batch" \
            "$OUTDIR/${model_name}_4n_task_batch_r${repeat}.jsonl" \
            $NUMA $TF_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON \
            --cpu-tp $TP --cpu-strategy taskflow --numa distribute
    done

    # ============================================================
    # 8 NUMA (all nodes), 192 threads, TP=8
    # ============================================================
    NUMA="numactl --cpunodebind=0,1,2,3,4,5,6,7 --membind=0,1,2,3,4,5,6,7"
    THREADS=192
    TP=8

    log ""
    log "====== 8NUMA, 192 threads, TP${TP} ======"

    for model_name in "${MODEL_ORDER[@]}"; do
        model_path="${MODELS[$model_name]}"
        log ""
        log "  Model: $model_name"

        # --- Decode ---
        run_bench "8N FJ-Pure decode" \
            "$OUTDIR/${model_name}_8n_fj_decode_r${repeat}.jsonl" \
            $NUMA $FJ_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 64 -npl 1 $COMMON --numa distribute

        run_bench "8N FJ+TP${TP} decode" \
            "$OUTDIR/${model_name}_8n_fjtp_decode_r${repeat}.jsonl" \
            $NUMA $FJ_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 64 -npl 1 $COMMON \
            --cpu-tp $TP --cpu-tp-runtime forkjoin --numa distribute

        run_bench "8N TaskInfer decode" \
            "$OUTDIR/${model_name}_8n_task_decode_r${repeat}.jsonl" \
            $NUMA $TF_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 64 -npl 1 $COMMON \
            --cpu-tp $TP --cpu-strategy taskflow --numa distribute

        # --- Prefill ---
        run_bench "8N FJ-Pure prefill" \
            "$OUTDIR/${model_name}_8n_fj_prefill_r${repeat}.jsonl" \
            $NUMA $FJ_BIN -m "$model_path" -t $THREADS \
            -npp 128,256,512 -ntg 1 -npl 1 $COMMON --numa distribute

        run_bench "8N FJ+TP${TP} prefill" \
            "$OUTDIR/${model_name}_8n_fjtp_prefill_r${repeat}.jsonl" \
            $NUMA $FJ_BIN -m "$model_path" -t $THREADS \
            -npp 128,256,512 -ntg 1 -npl 1 $COMMON \
            --cpu-tp $TP --cpu-tp-runtime forkjoin --numa distribute

        run_bench "8N TaskInfer prefill" \
            "$OUTDIR/${model_name}_8n_task_prefill_r${repeat}.jsonl" \
            $NUMA $TF_BIN -m "$model_path" -t $THREADS \
            -npp 128,256,512 -ntg 1 -npl 1 $COMMON \
            --cpu-tp $TP --cpu-strategy taskflow --numa distribute

        # --- Batched Decode ---
        run_bench "8N FJ-Pure batch" \
            "$OUTDIR/${model_name}_8n_fj_batch_r${repeat}.jsonl" \
            $NUMA $FJ_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON --numa distribute

        run_bench "8N FJ+TP${TP} batch" \
            "$OUTDIR/${model_name}_8n_fjtp_batch_r${repeat}.jsonl" \
            $NUMA $FJ_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON \
            --cpu-tp $TP --cpu-tp-runtime forkjoin --numa distribute

        run_bench "8N TaskInfer batch" \
            "$OUTDIR/${model_name}_8n_task_batch_r${repeat}.jsonl" \
            $NUMA $TF_BIN -m "$model_path" -t $THREADS \
            -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON \
            --cpu-tp $TP --cpu-strategy taskflow --numa distribute
    done
done

log ""
log "=========================================="
log "ALL EXPERIMENTS COMPLETE: $(date)"
log "Results in: $OUTDIR/"
log "=========================================="
