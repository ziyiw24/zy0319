#!/usr/bin/env bash
# Clipart1k 与 UODD：分别训练 15 epoch，只保留 epoch_15 并测试。
# 在 mmdetection 目录下执行：bash tools/run_clipart1k_uodd_experiments.sh
# 建议在 screen 中运行，并把本脚本输出重定向到日志文件。

set -e
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"

WORK_ROOT="${WORK_DIRS_ROOT:-/root/autodl-tmp/ETS/mmdetection_work_dirs}"
RESULTS_FILE="${1:-$(dirname "$0")/../clipart1k_uodd_results.md}"
[[ "$RESULTS_FILE" != /* ]] && RESULTS_FILE="$(cd "$(dirname "$0")/.." && pwd)/clipart1k_uodd_results.md"
RESULTS_FILE="$(cd "$(dirname "$RESULTS_FILE")" && pwd)/$(basename "$RESULTS_FILE")"

if ! grep -q "mAP (0.50:0.95)" "$RESULTS_FILE" 2>/dev/null; then
  echo "| Dataset   | Epoch | Date | mAP (0.50:0.95) | Checkpoint |" > "$RESULTS_FILE"
  echo "|-----------|-------|------|-----------------|------------|" >> "$RESULTS_FILE"
fi

cd "$(dirname "$0")/.."
DATE=$(date +%Y-%m-%d)

# ---------- Clipart1k ----------
echo "========== Clipart1k training =========="
CONFIG_C1K="configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-clipart1k.py"
WORK_C1K="${WORK_ROOT}/clipart1k"
mkdir -p "$WORK_C1K"
WORK_DIRS_ROOT="$WORK_C1K" bash tools/dist_train_muti.sh "$CONFIG_C1K" "0" 1

CKPT_DIR_C1K="${WORK_C1K}/exp1_gpu0"
for EP in 15; do
  if [ -f "$CKPT_DIR_C1K/epoch_${EP}.pth" ]; then
    echo "========== Clipart1k test (epoch_${EP}) =========="
    LOG=$(mktemp)
    bash tools/dist_test.sh "$CONFIG_C1K" "$CKPT_DIR_C1K/epoch_${EP}.pth" 1 2>&1 | tee "$LOG"
    MAP=$(grep 'IoU=0.50:0.95' "$LOG" | grep 'area=   all' | grep 'maxDets=100' | sed -n 's/.*= \([0-9.]*\) *$/\1/p' | head -1)
    rm -f "$LOG"
    echo "| clipart1k | $EP    | $DATE | $MAP | $CKPT_DIR_C1K/epoch_${EP}.pth |" >> "$RESULTS_FILE"
    echo "Clipart1k epoch_${EP} mAP (0.50:0.95): $MAP"
  fi
done

# ---------- UODD ----------
echo "========== UODD training =========="
CONFIG_UODD="configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-uodd.py"
WORK_UODD="${WORK_ROOT}/uodd"
mkdir -p "$WORK_UODD"
WORK_DIRS_ROOT="$WORK_UODD" bash tools/dist_train_muti.sh "$CONFIG_UODD" "0" 1

CKPT_DIR_UODD="${WORK_UODD}/exp1_gpu0"
for EP in 15; do
  if [ -f "$CKPT_DIR_UODD/epoch_${EP}.pth" ]; then
    echo "========== UODD test (epoch_${EP}) =========="
    LOG=$(mktemp)
    bash tools/dist_test.sh "$CONFIG_UODD" "$CKPT_DIR_UODD/epoch_${EP}.pth" 1 2>&1 | tee "$LOG"
    MAP=$(grep 'IoU=0.50:0.95' "$LOG" | grep 'area=   all' | grep 'maxDets=100' | sed -n 's/.*= \([0-9.]*\) *$/\1/p' | head -1)
    rm -f "$LOG"
    echo "| UODD      | $EP    | $DATE | $MAP | $CKPT_DIR_UODD/epoch_${EP}.pth |" >> "$RESULTS_FILE"
    echo "UODD epoch_${EP} mAP (0.50:0.95): $MAP"
  fi
done

echo "Done. Results appended to $RESULTS_FILE"
