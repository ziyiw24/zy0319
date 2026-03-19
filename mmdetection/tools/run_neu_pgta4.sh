#!/usr/bin/env bash
# NEU-DET 1/5/10-shot PGTA4（小框 blend + 自适应 EMA）：写入 neu_*_pgta4，不覆盖 pgta3
set -e
cd "$(dirname "$0")/.."
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"
BASE="${WORK_DIRS_BASE:-/root/autodl-tmp/ETS/mmdetection_work_dirs}"
mkdir -p "$BASE"

train_one() {
  local name=$1
  local config=$2
  local work_root="$BASE/$name"
  local log_file="$BASE/${name}_train.log"
  echo "========== TRAIN $name (work_dir=$work_root/exp1_gpu0) =========="
  WORK_DIRS_ROOT="$work_root" bash tools/dist_train_muti.sh "$config" 0 1 2>&1 | tee "$log_file"
  echo "========== Finished TRAIN $name =========="
}

test_one() {
  local name=$1
  local config=$2
  local ckpt="$BASE/$name/exp1_gpu0/epoch_15.pth"
  if [[ ! -f "$ckpt" ]]; then
    echo "WARN: $ckpt not found"
    if [[ -f "$BASE/$name/exp1_gpu0/last_checkpoint" ]]; then
      local f
      f=$(cat "$BASE/$name/exp1_gpu0/last_checkpoint" | tr -d '\n\r')
      [[ "$f" != /* ]] && ckpt="$BASE/$name/exp1_gpu0/$f" || ckpt="$f"
    fi
  fi
  local test_log="$BASE/${name}_test.log"
  echo "========== TEST $name checkpoint=$ckpt =========="
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$(pwd):$PYTHONPATH" \
  python tools/test.py "$config" "$ckpt" \
    --work-dir "$BASE/$name/exp1_gpu0" \
    2>&1 | tee "$test_log"
  echo "========== Finished TEST $name =========="
}

train_one "neu_1shot_pgta4"  "configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-neu-1shot.py"
train_one "neu_5shot_pgta4"  "configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-neu-5shot.py"
train_one "neu_10shot_pgta4" "configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-neu.py"

test_one "neu_1shot_pgta4"  "configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-neu-1shot.py"
test_one "neu_5shot_pgta4"  "configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-neu-5shot.py"
test_one "neu_10shot_pgta4" "configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB-neu.py"

echo "========== All NEU PGTA4 (1/5/10-shot) train+test done =========="
