#!/usr/bin/env bash
# PGTA2/PGTA3 实验：Clipart1k 5/10-shot、UODD 5/10-shot，权重与日志与旧实验区分存放
set -e
cd "$(dirname "$0")/.."
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"
BASE="${WORK_DIRS_BASE:-/root/autodl-tmp/ETS/mmdetection_work_dirs}"
mkdir -p "$BASE"

run_one() {
  local name=$1
  local config=$2
  local work_root="$BASE/$name"
  local log_file="$BASE/${name}_train.log"
  echo "========== $name (work_dir=$work_root) =========="
  WORK_DIRS_ROOT="$work_root" bash tools/dist_train_muti.sh "$config" 0 1 2>&1 | tee "$log_file"
  echo "========== Finished $name =========="
}

# 1) Clipart1k 5-shot
run_one "clipart1k_5shot_pgta2" "configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-clipart1k-5shot.py"

# 2) Clipart1k 10-shot
run_one "clipart1k_10shot_pgta2" "configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-clipart1k.py"

# 3) UODD 5-shot
run_one "uodd_5shot_pgta3" "configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-uodd-5shot.py"

# 4) UODD 10-shot
run_one "uodd_10shot_pgta3" "configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-uodd.py"

echo "========== All 4 PGTA experiments done =========="
