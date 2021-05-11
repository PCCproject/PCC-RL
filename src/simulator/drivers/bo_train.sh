#!/bin/bash
set -e
# save_dir=/tank/zxxia/PCC-RL/results_0503/bo_delay_new_no_delay_noise_fix_seed_fix_reward
save_dir=/tank/zxxia/PCC-RL/results_0503/bo_delay_new_no_delay_noise_fix_seed_fix_reward_10s_new
# save_dir=/tank/zxxia/PCC-RL/results_0503/bo_T_s
# save_dir=/tank/zxxia/PCC-RL/results_0503/bo_queue
# save_dir=/tank/zxxia/PCC-RL/results_0503/bo_bandwidth
model_path=/tank/zxxia/PCC-RL/results_0503/udr_7_dims/udr_large/seed_50/model_step_396000.ckpt
# model_path=/tank/zxxia/PCC-RL/results_0503/bo_delay/seed_10/bo_0/model_step_36000.ckpt


for seed in 10 20 30 40 50; do
CUDA_VISIBLE_DEVICES="" python train_with_bo.py --save-dir ${save_dir}/seed_${seed} \
    --model-path ${model_path} \
    --bo-interval 200000 --seed ${seed} &
done
