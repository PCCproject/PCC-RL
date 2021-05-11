#!/bin/bash

set -e

SAVE_DIR=udr_1_dim
seed=40
#rand_duration rand_bw rand_delay rand_loss rand_queue rand_bw_freq rand_delay_freq
for exp_name in rand_duration rand_bw rand_delay rand_loss rand_queue rand_bw_freq rand_delay_freq; do
    CUDA_VISIBLE_DEVICES="" python train_rl.py \
        --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed} \
        --exp-name ${exp_name} \
        --tensorboard-log aurora_tensorboard \
        --total-timesteps 5000000 \
        --delta-scale 1 \
        --randomization-range-file ../../config/train/udr_1_dim/${exp_name}.json \
        --seed ${seed} \
        --time-variant-bw &
done
