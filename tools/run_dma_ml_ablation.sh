#!/usr/bin/env bash
# Ablation across DMA fusion-weight predictor *model families* (not feature
# subsets -- see run_dma_gbm_ablation.sh for that axis). Trains each backend
# on the same data/features, then runs every checkpoint through
# tools/track_dma.py to get real HOTA/IDF1/MOTA numbers on MOT17 or SportMOT.
#
# Backends (fixed set):
#   mlp     yolox/DMA/train.py         DynamicWeightNet (torch), exact fused BCE loss
#   gbm     yolox/DMA/train_gbm.py     LightGBM,          exact fused BCE loss (custom objective)
#   xgb     yolox/DMA/tune_xgb.py      XGBoost,           exact fused BCE loss, Optuna BO over hyperparams
#   sk_rf       yolox/DMA/tune_sklearn.py --algo rf       RandomForestRegressor, MSE onto a
#                                                          closed-form per-sample target weight,
#                                                          Optuna BO over hyperparams
#   sk_logreg   yolox/DMA/tune_sklearn.py --algo logreg   LogisticRegression, predict_proba as w_motion,
#                                                          Optuna BO over hyperparams
# (sk_rf / sk_logreg use a different training signal than the exact fused
# BCE gbm/xgb/mlp optimise -- see model_sklearn.py. Treat this as "does the
# predictor family matter" rather than a perfectly controlled A/B.)
#
# xgb/sk_rf/sk_logreg are Bayesian-optimised (Optuna TPE, --n-trials trials)
# rather than trained once with hand-picked defaults, so the comparison isn't
# skewed by one backend getting a hyperparameter search and the others not --
# this mirrors how the reference gbm checkpoints (tuned_gbm_weights/, via
# tune_gbm.py) were produced. mlp is left as a single fixed-hyperparameter
# run; retune it with your own search if you also want it BO-tuned.
#
# Already have a checkpoint for a backend (e.g. mlp/gbm already trained)?
# Point CKPT_<algo> at it to skip training/tuning and go straight to tracking:
#   CKPT_mlp=dma_weights/dma_best_mot17.pth CKPT_gbm=dma_weights/dma_gbm_mot17.gbm \
#     tools/run_dma_ml_ablation.sh mot17
#
# Usage:
#   tools/run_dma_ml_ablation.sh mot17
#   tools/run_dma_ml_ablation.sh sportmot
#
# Env overrides:
#   ML_ALGOS (default "mlp gbm xgb sk_rf sk_logreg")
#   EPOCHS (mlp, default 50), NUM_BOOST_ROUND (gbm/xgb, default 500), EARLY_STOPPING (default 30)
#   N_TRIALS (xgb/sk_rf/sk_logreg Optuna trials, default 50), TUNE_TIMEOUT (seconds, optional per-backend budget)
#   FEATURE_INDICES (default "0,1,2,3,5"; set to "" to train on all features)
#   CKPT_<algo>=/path/to/checkpoint  -- skip training/tuning that backend, track with this checkpoint instead
#   FORCE_RETRAIN=1  -- ignore any already-finished checkpoint under OUT_ROOT and retrain everything
#
# Resumability: if a backend's default checkpoint (dma_weights/ml_ablation/<dataset>/<algo>/...)
# already exists from a previous run that got interrupted, training/tuning for that backend is
# skipped automatically and tracking runs straight away -- set FORCE_RETRAIN=1 to override.
set -uo pipefail

cd "$(dirname "$0")/.."

DATASET="${1:?Usage: $0 <mot17|sportmot>}"

case "$DATASET" in
  mot17)
    TRAIN_DATA_DIR="datasets/mot17_dma_train"
    EXP_FILE="exps/example/mot/yolox_x_mot17_half.py"
    CKPT="pretrained/bytetrack_x_mot17.pth.tar"
    FR_CONFIG="fast-reid/configs/MOT17/sbs_S50.yml"
    FR_WEIGHTS="reid_weights/mot17_sbs_S50.pth"
    ;;
  sportmot)
    TRAIN_DATA_DIR="datasets/sportmot_dma_train"
    EXP_FILE="exps/example/sportmot/yolox_x_sportsmot.py"
    CKPT="pretrained/yolox_x_sports_mix.pth.tar"
    FR_CONFIG="fast-reid/configs/SportMOT/sbs_S50.yml"
    FR_WEIGHTS="reid_weights/sportmot_sbs_S50.pth"
    ;;
  *)
    echo "Unknown dataset '${DATASET}' (expected mot17 or sportmot)" >&2
    exit 1
    ;;
esac

EPOCHS="${EPOCHS:-50}"
NUM_BOOST_ROUND="${NUM_BOOST_ROUND:-500}"
EARLY_STOPPING="${EARLY_STOPPING:-30}"
N_TRIALS="${N_TRIALS:-50}"
TUNE_TIMEOUT="${TUNE_TIMEOUT:-}"
ML_ALGOS="${ML_ALGOS:-mlp gbm xgb sk_rf sk_logreg}"
# Every backend trains on the same feature subset (FEATURE_INDICES) -- this
# script compares model families on a fixed feature set, not feature subsets.
# For that axis see run_dma_gbm_ablation.sh.
FEATURE_INDICES="${FEATURE_INDICES:-0,1,2,3,5}"
FEATURE_INDICES_ARGS=()
[ -n "$FEATURE_INDICES" ] && FEATURE_INDICES_ARGS=(--feature-indices "$FEATURE_INDICES")

OUT_ROOT="dma_weights/ml_ablation/${DATASET}"
LOG_DIR="YOLOX_outputs/ml_ablation_logs"
mkdir -p "$LOG_DIR"

COMMON_TRACK_ARGS=(-b 1 -d 1 --fuse --fp16 --save-videos
  --with-reid --fast-reid --fast-reid-config "$FR_CONFIG" --fast-reid-weights "$FR_WEIGHTS")

TUNE_TIMEOUT_ARGS=()
if [ -n "$TUNE_TIMEOUT" ]; then
  TUNE_TIMEOUT_ARGS=(--timeout "$TUNE_TIMEOUT")
fi

declare -a RESULTS=()
declare -a EXP_DIRS=()

for algo in $ML_ALGOS; do
  out_dir="${OUT_ROOT}/${algo}"
  train_log="${LOG_DIR}/train_${DATASET}_${algo}.log"
  track_log="${LOG_DIR}/track_${DATASET}_${algo}.log"
  exp_name="ml_${algo}_${DATASET}"

  override_var="CKPT_${algo}"
  override_ckpt="${!override_var:-}"

  case "$algo" in
    mlp)       default_ckpt_name="dma_best.pth";                ml_flag=() ;;
    gbm)       default_ckpt_name="dma_gbm_best.gbm";             ml_flag=(--ml gbm) ;;
    xgb)       default_ckpt_name="dma_xgb_tuned.xgb";            ml_flag=(--ml xgb) ;;
    sk_rf)     default_ckpt_name="dma_sklearn_rf_tuned.skl";     ml_flag=(--ml sklearn) ;;
    sk_logreg) default_ckpt_name="dma_sklearn_logreg_tuned.skl"; ml_flag=(--ml sklearn) ;;
    *)
      echo "Unknown algo '${algo}' in ML_ALGOS, skipping" >&2
      RESULTS+=("${algo}: UNKNOWN ALGO")
      continue
      ;;
  esac

  if [ -n "$override_ckpt" ]; then
    echo "=== [$(date '+%F %T')] [${DATASET}] '${algo}': using existing checkpoint ${override_ckpt} (skip training) ==="
    ckpt_path="$override_ckpt"
    train_status=0
  else
    ckpt_path="${out_dir}/${default_ckpt_name}"
    if [ -f "$ckpt_path" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
      echo "=== [$(date '+%F %T')] [${DATASET}] '${algo}': found existing checkpoint ${ckpt_path} from a previous run (skip training; set FORCE_RETRAIN=1 to redo) ==="
      train_status=0
    else
    case "$algo" in
      mlp)
        echo "=== [$(date '+%F %T')] [${DATASET}] Training '${algo}' -> ${out_dir} ==="
        python3 -m yolox.DMA.train \
          --data-dir "$TRAIN_DATA_DIR"  --out-dir "$out_dir" \
          --epochs "$EPOCHS" --loss bce \
          "${FEATURE_INDICES_ARGS[@]}" \
          > "$train_log" 2>&1
        train_status=$?
        ;;
      gbm)
        echo "=== [$(date '+%F %T')] [${DATASET}] Training '${algo}' -> ${out_dir} ==="
        python3 -m yolox.DMA.train_gbm \
          --data-dir "$TRAIN_DATA_DIR"  --out-dir "$out_dir" \
          --num-boost-round "$NUM_BOOST_ROUND" --early-stopping "$EARLY_STOPPING" \
          "${FEATURE_INDICES_ARGS[@]}" \
          > "$train_log" 2>&1
        train_status=$?
        ;;
      xgb)
        echo "=== [$(date '+%F %T')] [${DATASET}] Bayesian-tuning '${algo}' (${N_TRIALS} trials) -> ${out_dir} ==="
        python3 -m yolox.DMA.tune_xgb \
          --data-dir "$TRAIN_DATA_DIR"  --out-dir "$out_dir" \
          --n-trials "$N_TRIALS" --num-boost-round "$NUM_BOOST_ROUND" --early-stopping "$EARLY_STOPPING" \
          "${TUNE_TIMEOUT_ARGS[@]}" "${FEATURE_INDICES_ARGS[@]}" \
          > "$train_log" 2>&1
        train_status=$?
        ;;
      sk_rf)
        echo "=== [$(date '+%F %T')] [${DATASET}] Bayesian-tuning '${algo}' (${N_TRIALS} trials) -> ${out_dir} ==="
        python3 -m yolox.DMA.tune_sklearn \
          --data-dir "$TRAIN_DATA_DIR"  --out-dir "$out_dir" \
          --algo rf --n-trials "$N_TRIALS" "${TUNE_TIMEOUT_ARGS[@]}" "${FEATURE_INDICES_ARGS[@]}" \
          > "$train_log" 2>&1
        train_status=$?
        ;;
      sk_logreg)
        echo "=== [$(date '+%F %T')] [${DATASET}] Bayesian-tuning '${algo}' (${N_TRIALS} trials) -> ${out_dir} ==="
        python3 -m yolox.DMA.tune_sklearn \
          --data-dir "$TRAIN_DATA_DIR"  --out-dir "$out_dir" \
          --algo logreg --n-trials "$N_TRIALS" "${TUNE_TIMEOUT_ARGS[@]}" "${FEATURE_INDICES_ARGS[@]}" \
          > "$train_log" 2>&1
        train_status=$?
        ;;
    esac
    fi
  fi

  if [ "$train_status" -ne 0 ]; then
    echo "=== [$(date '+%F %T')] Train FAILED for '${algo}' (exit ${train_status}) - see ${train_log} - skipping tracking ===" >&2
    RESULTS+=("${algo}: TRAIN FAILED (exit ${train_status})")
    continue
  fi
  echo "=== [$(date '+%F %T')] Finished '${algo}' checkpoint: ${ckpt_path} ==="

  echo "=== [$(date '+%F %T')] [${DATASET}] Running tracking for '${algo}' -> YOLOX_outputs/${exp_name} ==="
  if [ "${#ml_flag[@]}" -gt 0 ]; then
    weight_args=("${ml_flag[@]}" --ml-weights "$ckpt_path")
  else
    weight_args=(--dma-weights "$ckpt_path")
  fi
  python3 tools/track_dma.py \
    -f "$EXP_FILE" -c "$CKPT" -expn "$exp_name" \
    "${COMMON_TRACK_ARGS[@]}" \
    "${weight_args[@]}" \
    > "$track_log" 2>&1
  track_status=$?
  if [ "$track_status" -eq 0 ]; then
    echo "=== [$(date '+%F %T')] Finished tracking '${algo}' OK (log: ${track_log}) ==="
    RESULTS+=("${algo}: OK")
    EXP_DIRS+=("YOLOX_outputs/${exp_name}")
  else
    echo "=== [$(date '+%F %T')] Tracking FAILED for '${algo}' (exit ${track_status}) - see ${track_log} - continuing ===" >&2
    RESULTS+=("${algo}: TRACK FAILED (exit ${track_status})")
  fi
done

echo ""
echo "=== Summary (${DATASET}) ==="
for r in "${RESULTS[@]}"; do
  echo "$r"
done

if [ "${#EXP_DIRS[@]}" -gt 0 ]; then
  echo ""
  echo "Aggregating metrics..."
  python3 tools/summarize_benchmark.py "${EXP_DIRS[@]}"
fi
