#!/usr/bin/env bash
# Run tools/track_dma.py for several tracking methods on SportMOT and save an
# annotated video per sequence for each run. Each method writes to its own
# experiment folder under YOLOX_outputs/, so results never overwrite each other.
set -uo pipefail

cd "$(dirname "$0")/.."

EXP_FILE="exps/example/sportmot/yolox_x_sportsmot.py"
CKPT="pretrained/yolox_x_sports_mix.pth.tar"
FR_CONFIG="fast-reid/configs/SportMOT/sbs_S50.yml"
FR_WEIGHTS="reid_weights/sportmot_sbs_S50.pth"
DMA_WEIGHTS="dma_weights/dma_best_sportmot.pth"
LTC_WEIGHTS="ltc_weights/ltc_motion_sportsmot.pth"

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

run_method "baseline_bytetrack"

run_method "with_reid_0_856" \
  --with-reid --fast-reid --fast-reid-config "$FR_CONFIG" --fast-reid-weights "$FR_WEIGHTS" --reid-weight 0.856

run_method "with_dma" \
  --with-reid --fast-reid --fast-reid-config "$FR_CONFIG" --fast-reid-weights "$FR_WEIGHTS" \
  --dma-weights "$DMA_WEIGHTS"

run_method "with_ltc" \
  --ltc-motion-ckpt "$LTC_WEIGHTS"

run_method "ltc_reid_0_856" \
  --ltc-motion-ckpt "$LTC_WEIGHTS" \
  --with-reid --fast-reid --fast-reid-config "$FR_CONFIG" --fast-reid-weights "$FR_WEIGHTS" --reid-weight 0.856

run_method "ltc_reid_dma" \
  --ltc-motion-ckpt "$LTC_WEIGHTS" \
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
  YOLOX_outputs/baseline_bytetrack \
  YOLOX_outputs/with_reid_0_856 \
  YOLOX_outputs/with_dma \
  YOLOX_outputs/with_ltc \
  YOLOX_outputs/ltc_reid_0_856 \
  YOLOX_outputs/ltc_reid_dma
