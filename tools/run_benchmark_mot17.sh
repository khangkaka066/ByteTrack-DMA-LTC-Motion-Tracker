#!/usr/bin/env bash
# Run tools/track_dma.py for several tracking methods on MOT17 and save an
# annotated video per sequence for each run. Each method writes to its own
# experiment folder under YOLOX_outputs/, so results never overwrite each other.
#
# Note: no LTC motion checkpoint is available for MOT17 (ltc_weights/ only has
# dancetrack and sportsmot variants), so LTC-based runs are omitted here.
set -uo pipefail

cd "$(dirname "$0")/.."

EXP_FILE="exps/example/mot/yolox_x_mot17_half.py"
CKPT="pretrained/bytetrack_x_mot17.pth.tar"
FR_CONFIG="fast-reid/configs/MOT17/sbs_S50.yml"
FR_WEIGHTS="reid_weights/mot17_sbs_S50.pth"
DMA_WEIGHTS="dma_weights/dma_best_mot17.pth"

LOG_DIR="YOLOX_outputs/benchmark_logs"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(-b 1 -d 1 --fuse --fp16 --save-videos)

declare -a RESULTS=()

run_method () {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  echo "=== [$(date '+%F %T')] Starting ${name} ==="

  python3 tools/track_dma.py \
    -f "$EXP_FILE" -c "$CKPT" -expn "$name" \
    "${COMMON_ARGS[@]}" "$@" \
    > "$log_file" 2>&1

  local status=$?
  if [ "$status" -eq 0 ]; then
    echo "=== [$(date '+%F %T')] Finished ${name} OK (log: ${log_file}) ==="
    RESULTS+=("${name}: OK")
  else
    echo "=== [$(date '+%F %T')] ${name} FAILED (exit ${status}) - see ${log_file} - continuing ===" >&2
    RESULTS+=("${name}: FAILED (exit ${status})")
  fi
}

run_method "baseline_bytetrack_mot17"

run_method "with_reid_0_856_mot17" \
  --with-reid --fast-reid --fast-reid-config "$FR_CONFIG" --fast-reid-weights "$FR_WEIGHTS" --reid-weight 0.856

run_method "with_dma_mot17" \
  --with-reid --fast-reid --fast-reid-config "$FR_CONFIG" --fast-reid-weights "$FR_WEIGHTS" \
  --dma-weights "$DMA_WEIGHTS"

echo ""
echo "=== Summary ==="
for r in "${RESULTS[@]}"; do
  echo "$r"
done

echo ""
echo "Aggregating metrics..."
python3 tools/summarize_benchmark.py \
  YOLOX_outputs/baseline_bytetrack_mot17 \
  YOLOX_outputs/with_reid_0_856_mot17 \
  YOLOX_outputs/with_dma_mot17
