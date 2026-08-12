#!/usr/bin/env bash
# Tune DynamicWeightGBM then DynamicWeightXGB on SportsMOT DMA data, back to
# back. tune_xgb.py only runs if tune_gbm.py succeeds; both write their own
# logs and checkpoints so the two runs never clobber each other.
set -uo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="datasets/sportmot_dma"
N_TRIALS="${N_TRIALS:-500}"
NUM_BOOST_ROUND="${NUM_BOOST_ROUND:-6000}"
EARLY_STOPPING="${EARLY_STOPPING:-50}"

GBM_OUT_DIR="tuned_gbm_weights/sportsmot"
XGB_OUT_DIR="tuned_xgb_weights/sportsmot"
LOG_DIR="YOLOX_outputs/tune_logs"
mkdir -p "$LOG_DIR"

echo "=== [$(date '+%F %T')] Starting tune_gbm (sportmot) -> ${GBM_OUT_DIR} ==="
python3 -m yolox.DMA.tune_gbm \
  --data-dir "$DATA_DIR" \
  --out-dir "$GBM_OUT_DIR" \
  --n-trials "$N_TRIALS" \
  --num-boost-round "$NUM_BOOST_ROUND" \
  --early-stopping "$EARLY_STOPPING" \
  > "${LOG_DIR}/tune_gbm_sportmot.log" 2>&1
gbm_status=$?

if [ "$gbm_status" -ne 0 ]; then
  echo "=== [$(date '+%F %T')] tune_gbm (sportmot) FAILED (exit ${gbm_status}) - see ${LOG_DIR}/tune_gbm_sportmot.log - skipping tune_xgb ===" >&2
  exit "$gbm_status"
fi
echo "=== [$(date '+%F %T')] Finished tune_gbm (sportmot) OK (log: ${LOG_DIR}/tune_gbm_sportmot.log) ==="

echo "=== [$(date '+%F %T')] Starting tune_xgb (sportmot) -> ${XGB_OUT_DIR} ==="
python3 -m yolox.DMA.tune_xgb \
  --data-dir "$DATA_DIR" \
  --out-dir "$XGB_OUT_DIR" \
  --n-trials "$N_TRIALS" \
  --num-boost-round "$NUM_BOOST_ROUND" \
  --early-stopping "$EARLY_STOPPING" \
  > "${LOG_DIR}/tune_xgb_sportmot.log" 2>&1
xgb_status=$?

if [ "$xgb_status" -ne 0 ]; then
  echo "=== [$(date '+%F %T')] tune_xgb (sportmot) FAILED (exit ${xgb_status}) - see ${LOG_DIR}/tune_xgb_sportmot.log ===" >&2
  exit "$xgb_status"
fi
echo "=== [$(date '+%F %T')] Finished tune_xgb (sportmot) OK (log: ${LOG_DIR}/tune_xgb_sportmot.log) ==="
