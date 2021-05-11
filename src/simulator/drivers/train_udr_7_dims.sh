#!/bin/bash

set -e

# SAVE_DIR=../../results_0415/udr_7_dims_fix_val_reward
# SAVE_DIR=../../results_0430/udr_7_dims
SAVE_DIR=../../results_0503/udr_7_dims
# SAVE_DIR=../../results_0430/udr_7_dims
# SAVE_DIR=tmp
# /../../results_0415/udr_7_dims
seed=50
#rand_duration rand_bw rand_delay rand_loss rand_queue rand_bw_freq rand_delay_freq
# for exp_name in range0 range1 range2; do
# for exp_name in range2_no_vary_bw; do
# range0_queue10
# for exp_name in 12mbps_queue50; do
#     CUDA_VISIBLE_DEVICES="" python train_rl.py \
#         --save-dir ${SAVE_DIR}/${exp_name} \
#         --exp-name ${exp_name}_seed_${seed} \
#         --tensorboard-log aurora_tensorboard \
#         --total-timesteps 5000000 \
#         --delta-scale 1 \
#         --randomization-range-file ../../config/train/udr_7_dims_0415/${exp_name}.json \
#         --seed ${seed} \
#         --time-variant-bw
# done
        # --pretrained-model-path ../../results_0415/udr_7_dims/range0_vary_bw_old/model_step_1576800.ckpt
for exp_name in udr_single; do
    for seed in 10 20 30 40 50; do
        CUDA_VISIBLE_DEVICES="" mpirun -np 2 python train_rl.py \
            --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed} \
            --exp-name ${exp_name}_seed_${seed} \
            --tensorboard-log aurora_tensorboard \
            --total-timesteps 5000000 \
            --delta-scale 1 \
            --randomization-range-file ../../config/train/udr_7_dims_0503/${exp_name}.json \
            --seed ${seed} \
            --time-variant-bw &
    done
done
