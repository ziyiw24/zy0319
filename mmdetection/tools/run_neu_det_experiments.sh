#!/usr/bin/env bash
# NEU-DET 1-shot and 10-shot: train (20 epoch), test, and record mAP.
# Run from mmdetection/: bash tools/run_neu_det_experiments.sh

set -e
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"

CONFIG="configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB.py"
WORK_ROOT="${WORK_DIRS_ROOT:-/root/autodl-tmp/ETS/mmdetection_work_dirs}"
RESULTS_FILE="${1:-$(dirname "$0")/../../NEU-DET_results.md}"
[[ "$RESULTS_FILE" != /* ]] && RESULTS_FILE="$(cd "$(dirname "$0")/../.." && pwd)/NEU-DET_results.md"
RESULTS_FILE="$(cd "$(dirname "$RESULTS_FILE")" && pwd)/$(basename "$RESULTS_FILE")"

# Ensure table exists
if ! grep -q "mAP (0.50:0.95)" "$RESULTS_FILE" 2>/dev/null; then
  echo "| Setting   | Date | mAP (0.50:0.95) | Checkpoint |" > "$RESULTS_FILE"
  echo "|-----------|------|-----------------|------------|" >> "$RESULTS_FILE"
fi

cd "$(dirname "$0")/.."
DATE=$(date +%Y-%m-%d)

# ---- 1-shot ----
echo "========== NEU-DET 1-shot training =========="
WORK_1SHOT="${WORK_ROOT}/neu_1shot"
mkdir -p "$WORK_1SHOT"
WORK_DIRS_ROOT="$WORK_1SHOT" bash tools/dist_train_muti.sh "$CONFIG" "0" 1 \
  --cfg-options train_dataloader.dataset.ann_file=annotations/1_shot.json

CKPT_1SHOT="${WORK_1SHOT}/exp1_gpu0"
if [ -f "$CKPT_1SHOT/epoch_20.pth" ]; then
  C1="$CKPT_1SHOT/epoch_20.pth"
else
  LAST=$(cat "$CKPT_1SHOT/last_checkpoint" 2>/dev/null | tr -d '\n\r')
  [ -n "$LAST" ] && C1="$CKPT_1SHOT/$LAST" || C1="$CKPT_1SHOT/epoch_20.pth"
fi
echo "========== NEU-DET 1-shot test =========="
LOG_1=$(mktemp)
bash tools/dist_test.sh "$CONFIG" "$C1" 1 2>&1 | tee "$LOG_1"
MAP_1=$(grep 'IoU=0.50:0.95' "$LOG_1" | grep 'area=   all' | grep 'maxDets=100' | sed -n 's/.*= \([0-9.]*\) *$/\1/p' | head -1)
rm -f "$LOG_1"
echo "| 1-shot    | $DATE | $MAP_1 | $C1 |" >> "$RESULTS_FILE"
echo "1-shot mAP (0.50:0.95): $MAP_1"

# ---- 10-shot ----
echo "========== NEU-DET 10-shot training =========="
WORK_10SHOT="${WORK_ROOT}/neu_10shot"
mkdir -p "$WORK_10SHOT"
WORK_DIRS_ROOT="$WORK_10SHOT" bash tools/dist_train_muti.sh "$CONFIG" "0" 1

CKPT_10SHOT="${WORK_10SHOT}/exp1_gpu0"
if [ -f "$CKPT_10SHOT/epoch_20.pth" ]; then
  C10="$CKPT_10SHOT/epoch_20.pth"
else
  LAST=$(cat "$CKPT_10SHOT/last_checkpoint" 2>/dev/null | tr -d '\n\r')
  [ -n "$LAST" ] && C10="$CKPT_10SHOT/$LAST" || C10="$CKPT_10SHOT/epoch_20.pth"
fi
echo "========== NEU-DET 10-shot test =========="
LOG_10=$(mktemp)
bash tools/dist_test.sh "$CONFIG" "$C10" 1 2>&1 | tee "$LOG_10"
MAP_10=$(grep 'IoU=0.50:0.95' "$LOG_10" | grep 'area=   all' | grep 'maxDets=100' | sed -n 's/.*= \([0-9.]*\) *$/\1/p' | head -1)
rm -f "$LOG_10"
echo "| 10-shot   | $DATE | $MAP_10 | $C10 |" >> "$RESULTS_FILE"
echo "10-shot mAP (0.50:0.95): $MAP_10"

echo "Done. Results appended to $RESULTS_FILE"
