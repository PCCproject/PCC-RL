#!/bin/bash

set -e

# SAVE_DIR=udr_4_dim_scale_0.01
# SAVE_DIR=udr_4_dims_correct_recv_rate
# SAVE_DIR=udr_4_dims_correct_recv_rate_vary_send_rate
# SAVE_DIR=udr_4_dims_log_queue
SAVE_DIR=udr_7_dims
# SAVE_DIR=udr_4_dim_scale_1

# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ../../results/${SAVE_DIR}/range0 \
#     --exp-name ${SAVE_DIR}_range0 \
#     --duration 10 \
#     --tensorboard-log aurora_tensorboard \
#     --bandwidth 2 2 --delay 50 50  --loss 0 0 --queue 10 10 \
#     --val-bandwidth 2  --val-delay 50 --val-loss 0 --val-queue 10 \
#     --total-timesteps 5000000 \
#     --delta-scale 1

# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ../../results/${SAVE_DIR}/range1 \
#     --exp-name ${SAVE_DIR}_range1 \
#     --duration 10 \
#     --tensorboard-log aurora_tensorboard \
#     --bandwidth 1.2 4 --delay 20 80 --loss 0 0.01 --queue 5 50 \
#     --val-bandwidth 2 --val-delay 50 --val-loss 0 --val-queue 10 \
#     --total-timesteps 5000000 \
#     --delta-scale 1

# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ../../results/${SAVE_DIR}/range2 \
#     --exp-name ${SAVE_DIR}_range2 \
#     --duration 10 \
#     --tensorboard-log aurora_tensorboard \
#     --bandwidth 0.6 6 --delay 5 100 --loss 0 0.02 --queue 1 100 \
#     --val-bandwidth 2 --val-delay 50 --val-loss 0 --val-queue 10 \
#     --total-timesteps 5000000 \
#     --delta-scale 1


# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ../../results/${SAVE_DIR}/range1_bo \
#     --exp-name ${SAVE_DIR}_range1_bo \
#     --duration 10 \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --pretrained-model-path ../../results/udr_4_dim_scale_1/range1/model_step_2865600.ckpt  \
#     --randomization-range-file ../../config/train/rand_4_dims_scale_1/range1.json

# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ../../results/${SAVE_DIR}/range1_bo_delay \
#     --exp-name ${SAVE_DIR}_range1_bo_delay \
#     --duration 10 \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --pretrained-model-path ../../results/udr_4_dim_scale_1/range1/model_step_2865600.ckpt  \
#     --randomization-range-file ../../config/train/rand_4_dims_scale_1/range1_delay.json
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ../../results/${SAVE_DIR}/range1_bo_queue \
#     --exp-name ${SAVE_DIR}_range1_bo_queue \
#     --duration 10 \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --pretrained-model-path ../../results/udr_4_dim_scale_1/range1/model_step_2865600.ckpt  \
#     --randomization-range-file ../../config/train/rand_4_dims_scale_1/range1_queue.json
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ../../results/${SAVE_DIR}/range1_bo_delay_28 \
#     --exp-name ${SAVE_DIR}_range1_bo_delay_28 \
#     --duration 10 \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --pretrained-model-path ../../results/udr_4_dim_scale_1/range1/model_step_2865600.ckpt  \
#     --randomization-range-file ../../config/train/rand_4_dims_scale_1/range1_delay.json

# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ../../results/${SAVE_DIR}/range1_bo_delay_73 \
#     --exp-name ${SAVE_DIR}_range1_bo_delay_73 \
#     --duration 10 \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --pretrained-model-path ../../results/udr_4_dim_scale_1/range1/model_step_2865600.ckpt  \
#     --randomization-range-file ../../config/train/rand_4_dims_scale_1/range1_delay.json



CUDA_VISIBLE_DEVICES="" python train_rl.py \
    --save-dir tmp \
    --exp-name ${SAVE_DIR}_range0 \
    --duration 10 \
    --tensorboard-log aurora_tensorboard \
    --total-timesteps 5000000 \
    --delta-scale 1 \
    --randomization-range-file ../../config/train/udr_4_dims_0327/range0.json
    # --save-dir ../../results_0326/${SAVE_DIR}/range0 \


# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ../../results_0326/${SAVE_DIR}/range1 \
#     --exp-name ${SAVE_DIR}_range1 \
#     --duration 10 \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_4_dims_0327/range1.json

# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ../../results_0326/${SAVE_DIR}/range2 \
#     --exp-name ${SAVE_DIR}_range2 \
#     --duration 10 \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_4_dims_0327/range2.json

# for config_id in 3 4 5 6 7; do
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ../../results_0326/${SAVE_DIR}/range${config_id} \
#     --exp-name ${SAVE_DIR}_range${config_id} \
#     --duration 10 \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_4_dims_0327/range${config_id}.json &
# done


# for config_id in 3; do
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ../../results_0326/${SAVE_DIR}/range${config_id} \
#     --exp-name ${SAVE_DIR}_range${config_id} \
#     --duration 10 \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims/range${config_id}.json \
#     --time-variant-bw
# done
# --save-dir ../../results_0326/${SAVE_DIR}/range${config_id} \
