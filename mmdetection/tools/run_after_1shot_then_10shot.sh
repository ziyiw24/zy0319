#!/usr/bin/env bash
# After 1-shot training finishes: run 1-shot test+record, then 10-shot train, then 10-shot test+record.
# Usage: bash tools/run_after_1shot_then_10shot.sh
# Optional: env WAIT_DIR=... CHECKPOINT_PATH=... to override paths.

set -e
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"

CONFIG="configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB.py"
WORK_ROOT="${WORK_DIRS_ROOT:-/root/autodl-tmp/ETS/mmdetection_work_dirs}"
CKPT_1SHOT="${WORK_ROOT}/neu_1shot/exp1_gpu0"
CKPT_10SHOT="${WORK_ROOT}/neu_10shot/exp1_gpu0"
RESULTS_FILE="$(cd "$(dirname "$0")/../.." && pwd)/NEU-DET_results.md"
DATE=$(date +%Y-%m-%d)

# Wait for 1-shot training to produce final checkpoint
C1=""
echo "Waiting for 1-shot training to produce epoch_20.pth or last_checkpoint..."
for i in $(seq 1 360); do
  if [ -f "$CKPT_1SHOT/epoch_20.pth" ]; then
    C1="$CKPT_1SHOT/epoch_20.pth"
    break
  fi
  if [ -f "$CKPT_1SHOT/last_checkpoint" ]; then
    LAST=$(cat "$CKPT_1SHOT/last_checkpoint" | tr -d '\n\r')
    if [ -n "$LAST" ] && [ -f "$CKPT_1SHOT/$LAST" ]; then
      C1="$CKPT_1SHOT/$LAST"
      break
    fi
  fi
  sleep 60
done
if [ -z "$C1" ]; then
  if [ -f "$CKPT_1SHOT/epoch_20.pth" ]; then C1="$CKPT_1SHOT/epoch_20.pth"; elif [ -f "$CKPT_1SHOT/last_checkpoint" ]; then L=$(cat "$CKPT_1SHOT/last_checkpoint" | tr -d '\n\r'); C1="$CKPT_1SHOT/$L"; fi
fi
[ -z "$C1" ] || [ ! -f "$C1" ] && { echo "No 1-shot checkpoint found."; exit 1; }
echo "1-shot checkpoint: $C1"

cd "$(dirname "$0")/.."

# 1-shot test and record
echo "========== NEU-DET 1-shot test =========="
LOG_1=$(mktemp)
bash tools/dist_test.sh "$CONFIG" "$C1" 1 2>&1 | tee "$LOG_1"
MAP_1=$(grep 'IoU=0.50:0.95' "$LOG_1" | grep 'area=   all' | grep 'maxDets=100' | sed -n 's/.*= \([0-9.]*\) *$/\1/p' | head -1)
rm -f "$LOG_1"
echo "| 1-shot    | $DATE | $MAP_1 | $C1 |" >> "$RESULTS_FILE"
echo "1-shot mAP (0.50:0.95): $MAP_1"

# 10-shot training
echo "========== NEU-DET 10-shot training =========="
WORK_DIRS_ROOT="${WORK_ROOT}/neu_10shot" bash tools/dist_train_muti.sh "$CONFIG" "0" 1

# Resolve 10-shot checkpoint
if [ -f "$CKPT_10SHOT/epoch_20.pth" ]; then C10="$CKPT_10SHOT/epoch_20.pth"; else C10="$CKPT_10SHOT/$(cat "$CKPT_10SHOT/last_checkpoint" 2>/dev/null | tr -d '\n\r')"; fi
echo "========== NEU-DET 10-shot test =========="
LOG_10=$(mktemp)
bash tools/dist_test.sh "$CONFIG" "$C10" 1 2>&1 | tee "$LOG_10"
MAP_10=$(grep 'IoU=0.50:0.95' "$LOG_10" | grep 'area=   all' | grep 'maxDets=100' | sed -n 's/.*= \([0-9.]*\) *$/\1/p' | head -1)
rm -f "$LOG_10"
echo "| 10-shot   | $DATE | $MAP_10 | $C10 |" >> "$RESULTS_FILE"
echo "10-shot mAP (0.50:0.95): $MAP_10"
echo "Done. Results in $RESULTS_FILE"
