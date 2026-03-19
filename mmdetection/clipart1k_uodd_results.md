| Dataset   | Epoch | Date | mAP (0.50:0.95) | Checkpoint |
|-----------|-------|------|-----------------|------------|
| clipart1k | 15    | 2026-03-03 | **0.544** | /root/autodl-tmp/ETS/mmdetection_work_dirs/clipart1k/exp1_gpu0/epoch_15.pth |
| clipart1k | 20    | 2026-03-03 | **0.543** | /root/autodl-tmp/ETS/mmdetection_work_dirs/clipart1k/exp1_gpu0/epoch_20.pth |
| UODD      | 15    | 2026-03-03 | 0.099 | /root/autodl-tmp/ETS/mmdetection_work_dirs/uodd/exp1_gpu0/epoch_15.pth |
| UODD      | 20    | 2026-03-03 | 0.099 | /root/autodl-tmp/ETS/mmdetection_work_dirs/uodd/exp1_gpu0/epoch_20.pth |

*Clipart1k 之前 0.237/0.238 为错误配置（test 仍用 NEU-DET）下的结果；修正 dataloader/evaluator 后重测为 0.543/0.544。*
| clipart1k | 15    | 2026-03-03 | 0.606 | /root/autodl-tmp/ETS/mmdetection_work_dirs/clipart1k/exp1_gpu0/epoch_15.pth |
| UODD      | 15    | 2026-03-03 | 0.310 | /root/autodl-tmp/ETS/mmdetection_work_dirs/uodd/exp1_gpu0/epoch_15.pth |
