#!/usr/bin/env bash
# Train DynamicWeightGBM for each feature-subset config from the
# analyze_features.py / sweep_gbm.py ablation study, then run every resulting
# checkpoint through tools/track_dma.py to get real HOTA/IDF1/MOTA numbers
# (not just the pair-level val_f1 proxy) on MOT17 or SportMOT.
#
# Requires fuse.py / model_gbm.py / train_gbm.py to carry feature_indices in
# the checkpoint's stats dict (see yolox/DMA/fuse.py DMAFusion) so a
# reduced-feature checkpoint can run through the real tracking pipeline.
#
# Feature subsets are given by name against yolox/DMA/features.py's FEAT_NAMES
# (currently: motion_cost, mahalanobis_norm, cov_trace_log, cosine_dist,
# bbox_area_log, tracklet_len_norm) via train_gbm.py --feature-indices
# 'drop:<name1>,<name2>' (see yolox/DMA/feature_spec.py) - resolved fresh
# against the current feature set, so this config list never goes stale the
# way hardcoded column indices from an older feature layout would.
#
# Usage:
#   tools/run_dma_gbm_ablation.sh mot17
#   tools/run_dma_gbm_ablation.sh sportmot
#
# Env overrides:
#   NUM_BOOST_ROUND (default 500), EARLY_STOPPING (default 30)
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

NUM_BOOST_ROUND="${NUM_BOOST_ROUND:-500}"
EARLY_STOPPING="${EARLY_STOPPING:-30}"

GBM_OUT_ROOT="dma_weights/gbm_ablation/${DATASET}"
LOG_DIR="YOLOX_outputs/gbm_ablation_logs"
mkdir -p "$LOG_DIR"

# config name -> --feature-indices spec ("" = all features = baseline). One
# leave-one-out config per current feature, generated from
# yolox/DMA/features.py FEAT_NAMES - edit FEAT_NAMES there (not here) if the
# feature set changes, and these configs stay correct.

declare -A CONFIGS=(
  [baseline]=""
  [drop_motion_cost]="drop:motion_cost"
  [drop_mahalanobis_norm]="drop:mahalanobis_norm"
  [drop_cov_trace_log]="drop:cov_trace_log"
  [drop_cosine_dist]="drop:cosine_dist"
  [drop_bbox_area_log]="drop:bbox_area_log"
  [drop_tracklet_len_norm]="drop:tracklet_len_norm"
  [only_motion_appearance_cost]="motion_cost,cosine_dist"
)
CONFIG_ORDER=(baseline drop_motion_cost drop_mahalanobis_norm drop_cov_trace_log drop_cosine_dist drop_bbox_area_log drop_tracklet_len_norm only_motion_appearance_cost)

COMMON_TRACK_ARGS=(-b 1 -d 1 --fuse --fp16 --save-videos
  --with-reid --fast-reid --fast-reid-config "$FR_CONFIG" --fast-reid-weights "$FR_WEIGHTS")

declare -a RESULTS=()
declare -a EXP_DIRS=()

for name in "${CONFIG_ORDER[@]}"; do
  idx="${CONFIGS[$name]}"
  out_dir="${GBM_OUT_ROOT}/${name}"
  ckpt_path="${out_dir}/dma_gbm_best.gbm"
  train_log="${LOG_DIR}/train_${DATASET}_${name}.log"
  track_log="${LOG_DIR}/track_${DATASET}_${name}.log"
  exp_name="gbm_${name}_${DATASET}"

  echo "=== [$(date '+%F %T')] [${DATASET}] Training GBM config '${name}' (feature_indices=${idx:-all}) -> ${out_dir} ==="
  train_args=(--data-dir "$TRAIN_DATA_DIR" --out-dir "$out_dir"
    --num-boost-round "$NUM_BOOST_ROUND" --early-stopping "$EARLY_STOPPING")
  [ -n "$idx" ] && train_args+=(--feature-indices "$idx")

  python3 -m yolox.DMA.train_gbm "${train_args[@]}" > "$train_log" 2>&1
  train_status=$?
  if [ "$train_status" -ne 0 ]; then
    echo "=== [$(date '+%F %T')] Train FAILED for '${name}' (exit ${train_status}) - see ${train_log} - skipping tracking ===" >&2
    RESULTS+=("${name}: TRAIN FAILED (exit ${train_status})")
    continue
  fi
  echo "=== [$(date '+%F %T')] Finished training '${name}' OK (log: ${train_log}) ==="

  echo "=== [$(date '+%F %T')] [${DATASET}] Running tracking for '${name}' -> YOLOX_outputs/${exp_name} ==="
  python3 tools/track_dma.py \
    -f "$EXP_FILE" -c "$CKPT" -expn "$exp_name" \
    "${COMMON_TRACK_ARGS[@]}" \
    --ml gbm --ml-weights "$ckpt_path" \
    > "$track_log" 2>&1
  track_status=$?
  if [ "$track_status" -eq 0 ]; then
    echo "=== [$(date '+%F %T')] Finished tracking '${name}' OK (log: ${track_log}) ==="
    RESULTS+=("${name}: OK")
    EXP_DIRS+=("YOLOX_outputs/${exp_name}")
  else
    echo "=== [$(date '+%F %T')] Tracking FAILED for '${name}' (exit ${track_status}) - see ${track_log} - continuing ===" >&2
    RESULTS+=("${name}: TRACK FAILED (exit ${track_status})")
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
