#!/usr/bin/env bash

# 修复 RTX 4090 等显卡上 nvrtc invalid -arch 报错：用 8.6+PTX 做 JIT 编译
export TORCH_CUDA_ARCH_LIST="8.6+PTX"
mkdir -p "${HOME:-/root}/.cache/torch/kernels" 2>/dev/null || true

CONFIG=$1
GPU_IDS=${2:-"0,1,2,3"}
NUM_RUNS=${3:-5}  # 要运行的总轮数，每轮4个任务

# work_dirs 写入可写数据目录（AutoDL 下为 autodl-tmp，可通过 WORK_DIRS_ROOT 覆盖）
WORK_DIRS_ROOT="${WORK_DIRS_ROOT:-/root/autodl-tmp/ETS/mmdetection_work_dirs}"
mkdir -p "$WORK_DIRS_ROOT"
# 临时目录也放到数据盘，避免 save_checkpoint 时占满系统盘
export TMPDIR="${TMPDIR:-/root/autodl-tmp/ETS/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR"

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

for RUN_ID in $(seq 1 $NUM_RUNS); do
  echo "========== Starting Experiment Group $RUN_ID =========="

  for ((i = 0; i < NUM_GPUS; i++)); do
    GPU_ID=${GPU_ARRAY[$i]}
    echo "  Launching task on GPU $GPU_ID (Experiment $RUN_ID)"

    OUTPUT_DIR="${WORK_DIRS_ROOT}/exp${RUN_ID}_gpu${GPU_ID}"
    mkdir -p "$OUTPUT_DIR"
    LOG_FILE="${OUTPUT_DIR}/train.log"

    # 若传入 --resume，将 last_checkpoint 转为绝对路径，避免 mmengine 在 cwd 下找不到
    EXTRA_ARGS=("${@:4}")
    for i in "${!EXTRA_ARGS[@]}"; do
      if [[ "${EXTRA_ARGS[$i]}" == "--resume" ]]; then
        next=$((i+1))
        if [[ -z "${EXTRA_ARGS[$next]:-}" || "${EXTRA_ARGS[$next]}" == --* ]]; then
          if [[ -f "$OUTPUT_DIR/last_checkpoint" ]]; then
            CKPT=$(cat "$OUTPUT_DIR/last_checkpoint" | tr -d '\n\r')
            [[ "$CKPT" != /* ]] && CKPT="$OUTPUT_DIR/$CKPT"
            EXTRA_ARGS[$i]="--resume"
            EXTRA_ARGS=("${EXTRA_ARGS[@]:0:$next}" "$CKPT" "${EXTRA_ARGS[@]:$next}")
          fi
        fi
        break
      fi
    done

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}" \
    TMPDIR="$TMPDIR" TEMP="$TMPDIR" TMP="$TMPDIR" \
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python $(dirname "$0")/train.py $CONFIG \
      --work-dir "$OUTPUT_DIR" \
      --launcher none "${EXTRA_ARGS[@]}" \
      2>&1 | tee "$LOG_FILE"

  done

  # 等待这一组全部跑完
  wait
  echo "========== Finished Experiment Group $RUN_ID =========="
done
