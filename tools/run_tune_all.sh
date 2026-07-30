#!/usr/bin/env bash
# Run tune_reid_weight.py sequentially over MOT17 / SportsMOT / DanceTrack.
# One dataset failing does not stop the others -- each run is logged separately
# and its exit status is summarized at the end.
set -uo pipefail

cd "$(dirname "$0")/.."

N_TRIALS="${N_TRIALS:-20}"
N_STARTUP="${N_STARTUP:-5}"
PATIENCE="${PATIENCE:-8}"
LOG_DIR="YOLOX_outputs/tune_logs"
mkdir -p "$LOG_DIR"

# name | exp_file | ckpt | fast_reid_config | fast_reid_weights | extra_track_args
declare -a DATASETS=(
  # "mot17|exps/example/mot/yolox_x_mot17_half.py|pretrained/bytetrack_x_mot17.pth.tar|fast-reid/configs/MOT17/sbs_S50.yml|reid_weights/mot17_sbs_S50.pth"
  "sportmot|exps/example/sportmot/yolox_x_sportsmot.py|pretrained/yolox_x_sports_mix.pth.tar|fast-reid/configs/SportMOT/sbs_S50.yml|reid_weights/sportmot_sbs_S50.pth"
  "dancetrack|exps/example/dancetrack/yolox_x_dancetrack.py|pretrained/bytetrack_dancetrack.pth.tar|fast-reid/configs/Dancetrack/sbs_S50.yml|reid_weights/dancetrack_sbs_S50.pth"
)

declare -a RESULTS=()

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r name exp_file ckpt fr_config fr_weights <<< "$entry"
  study_name="tune_${name}"
  log_file="${LOG_DIR}/${study_name}.log"

  echo "=== [$(date '+%F %T')] Starting ${name} -> study=${study_name} ==="

  python3 tools/tune_reid_weight.py \
    -f "$exp_file" \
    -c "$ckpt" \
    --study-name "$study_name" \
    --n-trials "$N_TRIALS" \
    --n-startup-trials "$N_STARTUP" \
    --patience "$PATIENCE" \
    -- -b 1 -d 1 --fuse --fp16 \
       --fast-reid \
       --fast-reid-config "$fr_config" \
       --fast-reid-weights "$fr_weights" \
    > "$log_file" 2>&1

  status=$?
  if [ "$status" -eq 0 ]; then
    echo "=== [$(date '+%F %T')] Finished ${name} OK (log: ${log_file}) ==="
    RESULTS+=("${name}: OK")
  else
    echo "=== [$(date '+%F %T')] ${name} FAILED (exit ${status}) - see ${log_file} - continuing to next dataset ===" >&2
    RESULTS+=("${name}: FAILED (exit ${status})")
  fi
done

echo ""
echo "=== Summary ==="
for r in "${RESULTS[@]}"; do
  echo "$r"
done
