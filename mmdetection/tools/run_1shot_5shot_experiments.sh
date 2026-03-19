#!/usr/bin/env bash
# 1-shot 与 5-shot：Clipart1k 与 UODD 各训练 15 epoch，保留 epoch_15 并测试。
# 在 mmdetection 目录下执行：bash tools/run_1shot_5shot_experiments.sh
# 建议在 screen 中运行：screen -S 1shot5shot -dm bash -c "cd /root/ETS/mmdetection && bash tools/run_1shot_5shot_experiments.sh 2>&1 | tee train_1shot_5shot.log"

set -e
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"

WORK_ROOT="${WORK_DIRS_ROOT:-/root/autodl-tmp/ETS/mmdetection_work_dirs}"
RESULTS_FILE="${1:-$(dirname "$0")/../clipart1k_uodd_1shot_5shot_results.md}"
[[ "$RESULTS_FILE" != /* ]] && RESULTS_FILE="$(cd "$(dirname "$0")/.." && pwd)/clipart1k_uodd_1shot_5shot_results.md"
RESULTS_FILE="$(cd "$(dirname "$RESULTS_FILE")" && pwd)/$(basename "$RESULTS_FILE")"

if ! grep -q "mAP (0.50:0.95)" "$RESULTS_FILE" 2>/dev/null; then
  echo "| Dataset   | Shot | Epoch | Date | mAP (0.50:0.95) | Checkpoint |" > "$RESULTS_FILE"
  echo "|-----------|------|-------|------|-----------------|------------|" >> "$RESULTS_FILE"
fi

cd "$(dirname "$0")/.."
DATE=$(date +%Y-%m-%d)

run_one() {
  local CONFIG="$1"
  local WORK_SUFFIX="$2"
  local DATASET_NAME="$3"
  local SHOT="$4"
  local WORK_DIR="${WORK_ROOT}/${WORK_SUFFIX}"
  mkdir -p "$WORK_DIR"
  echo "========== $DATASET_NAME ${SHOT}-shot training =========="
  WORK_DIRS_ROOT="$WORK_DIR" bash tools/dist_train_muti.sh "$CONFIG" "0" 1
  local CKPT_DIR="${WORK_DIR}/exp1_gpu0"
  if [ -f "$CKPT_DIR/epoch_15.pth" ]; then
    echo "========== $DATASET_NAME ${SHOT}-shot test (epoch_15) =========="
    LOG=$(mktemp)
    bash tools/dist_test.sh "$CONFIG" "$CKPT_DIR/epoch_15.pth" 1 2>&1 | tee "$LOG"
    MAP=$(grep 'IoU=0.50:0.95' "$LOG" | grep 'area=   all' | grep 'maxDets=100' | sed -n 's/.*= \([0-9.]*\) *$/\1/p' | head -1)
    rm -f "$LOG"
    echo "| $DATASET_NAME | $SHOT | 15    | $DATE | $MAP | $CKPT_DIR/epoch_15.pth |" >> "$RESULTS_FILE"
    echo "$DATASET_NAME ${SHOT}-shot epoch_15 mAP (0.50:0.95): $MAP"
  fi
}

# Clipart1k 1-shot & 5-shot
run_one "configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-clipart1k-1shot.py" "clipart1k_1shot" "clipart1k" "1"
run_one "configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-clipart1k-5shot.py" "clipart1k_5shot" "clipart1k" "5"

# UODD 1-shot & 5-shot
run_one "configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-uodd-1shot.py" "uodd_1shot" "UODD" "1"
run_one "configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-uodd-5shot.py" "uodd_5shot" "UODD" "5"

echo "Done. Results appended to $RESULTS_FILE"
