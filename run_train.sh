#!/usr/bin/env bash
# 单卡训练：GPU 0，共 1 组实验
# 使用方式：在 ETS 根目录执行 ./run_train.sh
cd "$(dirname "$0")/mmdetection"
./tools/dist_train_muti.sh configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB.py "0" 1
