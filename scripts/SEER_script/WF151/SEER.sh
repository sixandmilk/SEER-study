#!/usr/bin/env bash

set -u
set -o pipefail

# Allow this script to be launched from any directory.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR" || exit 1

export CUDA_VISIBLE_DEVICES=0

mkdir -p logs/WF151_SEER

DATA_NAME="WF151.csv"
MODEL_NAME="seer.SEER"
SAVE_PATH="WF151/SEER-base"
GPU_ID=0
NUM_TEST=1440

run_one() {
    local horizon="$1"
    local log_file="logs/WF151_SEER/horizon_${horizon}.log"
    local model_hyper_params

    model_hyper_params=$(printf \
        '{"batch_size": 16, "d_ff": 512, "d_model": 256, "d_score": 32, "dropout": 0.1, "e_layers": 2, "head_dropout": 0, "pred_len": %s, "k": 2, "loss": "MAE", "lr": 0.0001, "lradj": "type3", "n_heads": 8, "norm": true, "num_epochs": 30, "num_experts": 4, "patch_len": 16, "patience": 5, "scaling": 1, "score_threshold": 0.5, "seq_len": 96, "stride": 8}' \
        "$horizon")

    echo "[$(date '+%F %T')] start horizon=${horizon}; log=${log_file}"

    python -u ./scripts/run_benchmark.py \
        --config-path "rolling_forecast_config.json" \
        --data-name-list "$DATA_NAME" \
        --strategy-args "{\"horizon\": ${horizon}, \"num_test\": ${NUM_TEST}}" \
        --model-name "$MODEL_NAME" \
        --model-hyper-params "$model_hyper_params" \
        --gpus "$GPU_ID" \
        --num-workers 1 \
        --timeout 60000 \
        --save-path "$SAVE_PATH" \
        > "$log_file" 2>&1

    local status=$?
    echo "[$(date '+%F %T')] finish horizon=${horizon}; exit=${status}" | tee -a "$log_file"
    return "$status"
}

# Run horizons sequentially so only one experiment uses GPU 0 at a time.
HORIZONS=(8 16 20)
FAILED=0

for horizon in "${HORIZONS[@]}"; do
    if ! run_one "$horizon"; then
        FAILED=1
    fi
done

exit "$FAILED"
