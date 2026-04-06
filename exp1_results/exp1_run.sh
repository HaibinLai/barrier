#!/bin/bash
# =================================================================
# exp1_run.sh — Experiment 1: Overall Performance Comparison
# 3 Systems × 3 Models × 3 NUMA configs × Decode + Prefill
# =================================================================
set -euo pipefail

# ======================== Configuration ==========================
FJ_BIN=/home/haibin/tp/fork-join/build/bin/llama-batched-bench
TF_BIN=/home/haibin/tp/tp-llama/build/bin/llama-batched-bench

OUTDIR=/home/haibin/tp/exp1_results/raw
mkdir -p "$OUTDIR"

# Models
declare -A MODELS
MODELS[qwen3_4b]="$HOME/model/Qwen3-4B-Ins-2507-f16.gguf"
MODELS[llama3_8b]="$HOME/model/Meta-Llama-3.1-8B-Instruct.FP16.gguf"
MODELS[qwen25_14b]="$HOME/model/Qwen2.5-14B-Instruct-f16.gguf"

MODEL_ORDER=(qwen3_4b llama3_8b qwen25_14b)

# NUMA configurations: name -> "cpunodebind membind threads tp_degree"
declare -A NUMA_CFGS
NUMA_CFGS[1n]="0 0 24 1"
NUMA_CFGS[4n]="0,1,2,3 0,1,2,3 96 4"
NUMA_CFGS[8n]="0,1,2,3,4,5,6,7 0,1,2,3,4,5,6,7 192 8"

NUMA_ORDER=(1n 4n 8n)

REPEATS=3
COMMON="-c 4096 -b 512 -ub 256 --output-format jsonl"

# ======================== Functions ==============================

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUTDIR/exp1.log"
}

run_single() {
    local label="$1"
    local outfile="$2"
    shift 2
    local cmd=("$@")

    log "  Running: $label"
    log "  CMD: ${cmd[*]}"

    # Run and capture output
    if "${cmd[@]}" > "$outfile" 2>&1; then
        log "  Done: $label"
    else
        log "  FAILED: $label (exit $?)"
    fi
}

build_numa_prefix() {
    local nodes="$1"
    local membind="$2"
    echo "numactl --cpunodebind=$nodes --membind=$membind"
}

# ======================== Main Loop ==============================

log "=========================================="
log "Experiment 1: Overall Performance"
log "Start: $(date)"
log "=========================================="

for numa_name in "${NUMA_ORDER[@]}"; do
    read -r nodes membind threads tp <<< "${NUMA_CFGS[$numa_name]}"
    NUMA_PREFIX="numactl --cpunodebind=$nodes --membind=$membind"

    log ""
    log "============ NUMA: ${numa_name} (${threads}t, TP${tp}) ============"

    for model_name in "${MODEL_ORDER[@]}"; do
        model_path="${MODELS[$model_name]}"

        # Skip Qwen2.5-14B if not downloaded yet
        if [[ "$model_name" == "qwen25_14b" ]] && [[ ! -f "$model_path" ]]; then
            log "  SKIP: $model_name (model not found: $model_path)"
            continue
        fi

        log ""
        log "  Model: $model_name"

        for repeat in $(seq 1 $REPEATS); do
            log ""
            log "  --- Repeat $repeat/$REPEATS ---"

            # ============ DECODE Stage ============
            # Test token generation latency (memory-bound)
            DECODE_ARGS="-npp 64 -ntg 64 -npl 1 $COMMON"

            # Group A: Fork-Join Pure (no TP)
            outfile="$OUTDIR/${model_name}_${numa_name}_fj_decode_r${repeat}.jsonl"
            if [[ "$numa_name" == "1n" ]]; then
                run_single "FJ-Pure decode" "$outfile" \
                    $NUMA_PREFIX $FJ_BIN -m "$model_path" -t "$threads" $DECODE_ARGS \
                    --numa distribute
            else
                run_single "FJ-Pure decode" "$outfile" \
                    $NUMA_PREFIX $FJ_BIN -m "$model_path" -t "$threads" $DECODE_ARGS \
                    --numa distribute
            fi

            # Group B: Fork-Join + TP
            if [[ "$tp" -gt 1 ]]; then
                outfile="$OUTDIR/${model_name}_${numa_name}_fjtp_decode_r${repeat}.jsonl"
                run_single "FJ+TP${tp} decode" "$outfile" \
                    $NUMA_PREFIX $FJ_BIN -m "$model_path" -t "$threads" $DECODE_ARGS \
                    --cpu-tp "$tp" --cpu-tp-runtime forkjoin --numa distribute
            fi

            # Group C: TaskInfer (Task + TP)
            if [[ "$tp" -gt 1 ]]; then
                outfile="$OUTDIR/${model_name}_${numa_name}_task_decode_r${repeat}.jsonl"
                run_single "TaskInfer decode" "$outfile" \
                    $NUMA_PREFIX $TF_BIN -m "$model_path" -t "$threads" $DECODE_ARGS \
                    --cpu-tp "$tp" --cpu-strategy taskflow --numa distribute
            else
                # 1N: taskflow without TP
                outfile="$OUTDIR/${model_name}_${numa_name}_task_decode_r${repeat}.jsonl"
                run_single "TaskInfer decode" "$outfile" \
                    $NUMA_PREFIX $TF_BIN -m "$model_path" -t "$threads" $DECODE_ARGS \
                    --cpu-strategy taskflow --numa distribute
            fi

            # ============ PREFILL Stage ============
            PREFILL_ARGS="-npp 128,256,512 -ntg 1 -npl 1 $COMMON"

            # Group A: Fork-Join Pure
            outfile="$OUTDIR/${model_name}_${numa_name}_fj_prefill_r${repeat}.jsonl"
            run_single "FJ-Pure prefill" "$outfile" \
                $NUMA_PREFIX $FJ_BIN -m "$model_path" -t "$threads" $PREFILL_ARGS \
                --numa distribute

            # Group B: Fork-Join + TP
            if [[ "$tp" -gt 1 ]]; then
                outfile="$OUTDIR/${model_name}_${numa_name}_fjtp_prefill_r${repeat}.jsonl"
                run_single "FJ+TP${tp} prefill" "$outfile" \
                    $NUMA_PREFIX $FJ_BIN -m "$model_path" -t "$threads" $PREFILL_ARGS \
                    --cpu-tp "$tp" --cpu-tp-runtime forkjoin --numa distribute
            fi

            # Group C: TaskInfer
            if [[ "$tp" -gt 1 ]]; then
                outfile="$OUTDIR/${model_name}_${numa_name}_task_prefill_r${repeat}.jsonl"
                run_single "TaskInfer prefill" "$outfile" \
                    $NUMA_PREFIX $TF_BIN -m "$model_path" -t "$threads" $PREFILL_ARGS \
                    --cpu-tp "$tp" --cpu-strategy taskflow --numa distribute
            else
                outfile="$OUTDIR/${model_name}_${numa_name}_task_prefill_r${repeat}.jsonl"
                run_single "TaskInfer prefill" "$outfile" \
                    $NUMA_PREFIX $TF_BIN -m "$model_path" -t "$threads" $PREFILL_ARGS \
                    --cpu-strategy taskflow --numa distribute
            fi

            # ============ BATCHED DECODE Stage ============
            BATCH_ARGS="-npp 64 -ntg 32 -npl 1,4,8,16 $COMMON"

            # Group A: Fork-Join Pure
            outfile="$OUTDIR/${model_name}_${numa_name}_fj_batch_r${repeat}.jsonl"
            run_single "FJ-Pure batch" "$outfile" \
                $NUMA_PREFIX $FJ_BIN -m "$model_path" -t "$threads" $BATCH_ARGS \
                --numa distribute

            # Group B: Fork-Join + TP
            if [[ "$tp" -gt 1 ]]; then
                outfile="$OUTDIR/${model_name}_${numa_name}_fjtp_batch_r${repeat}.jsonl"
                run_single "FJ+TP${tp} batch" "$outfile" \
                    $NUMA_PREFIX $FJ_BIN -m "$model_path" -t "$threads" $BATCH_ARGS \
                    --cpu-tp "$tp" --cpu-tp-runtime forkjoin --numa distribute
            fi

            # Group C: TaskInfer
            if [[ "$tp" -gt 1 ]]; then
                outfile="$OUTDIR/${model_name}_${numa_name}_task_batch_r${repeat}.jsonl"
                run_single "TaskInfer batch" "$outfile" \
                    $NUMA_PREFIX $TF_BIN -m "$model_path" -t "$threads" $BATCH_ARGS \
                    --cpu-tp "$tp" --cpu-strategy taskflow --numa distribute
            else
                outfile="$OUTDIR/${model_name}_${numa_name}_task_batch_r${repeat}.jsonl"
                run_single "TaskInfer batch" "$outfile" \
                    $NUMA_PREFIX $TF_BIN -m "$model_path" -t "$threads" $BATCH_ARGS \
                    --cpu-strategy taskflow --numa distribute
            fi
        done
    done
done

log ""
log "=========================================="
log "All experiments complete: $(date)"
log "Results in: $OUTDIR/"
log "=========================================="
