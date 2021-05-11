#!/bin/bash

set -e

# CUDA_VISIBLE_DEVICES="" python train_with_bo.py \
#     --config-file ../../config/train/udr_7_dims_0415/range0.json \
#     --save-dir tmp
# save_dir=test_bo_0421
# seed=20

# exp_name=delay
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_bo_delay0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims/range0/model_step_2865600.ckpt \
#     --seed ${seed} \
#     --time-variant-bw

# exp_name=duration
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_bo_duration0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims/range0/model_step_2865600.ckpt \
#     --seed ${seed} \
#     --time-variant-bw


# exp_name=T_s_delay
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_bo_T_s_delay0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims/range0/model_step_2865600.ckpt \
#     --seed ${seed} \
#     --time-variant-bw



# exp_name=ack_delay_prob
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_bo_ack_delay_prob0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims/range0/model_step_2865600.ckpt \
#     --seed ${seed} \
#     --time-variant-bw







# ------------------------------------------------
# exp_name=bo_delay
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#     --seed ${seed} \
#     --time-variant-bw

# exp_name=bo_T_s_delay
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#     --seed ${seed} \
#     --time-variant-bw



# exp_name=bo_T_s_bandwidth
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#     --seed ${seed} \
#     --time-variant-bw


# exp_name=bo_bandwidth
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#     --seed ${seed} \
#     --time-variant-bw

# exp_name=bo_bandwidth_11_12_single_env
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#     --seed ${seed} \
#     --time-variant-bw

# exp_name=bo_bandwidth_10_12_single_env
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name}_linear_0.001_12_batch720 \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#     --seed ${seed} \
#     --time-variant-bw


# exp_name=bo_bandwidth_12_only
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#     --seed ${seed} \
#     --time-variant-bw

# exp_name=bo_bandwidth_single_env
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#     --seed ${seed} \
#     --time-variant-bw

# exp_name=bo_bandwidth_single_env
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name}_linear \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 3000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#     --seed ${seed} \
#     --time-variant-bw

# exp_name=bo_T_s_bandwidth_narrow
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#     --seed ${seed} \
#     --time-variant-bw

# exp_name=bo_T_s_bandwidth_3_4_logscale_small_bw
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#     --seed ${seed} \
#     --time-variant-bw &
#
# exp_name=bo_T_s_bandwidth_3_4_logscale
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#     --seed ${seed} \
#     --time-variant-bw &

# exp_name=bo_T_s_bandwidth1
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}.json \
#     --pretrained-model-path test_bo_0421/bo_T_s_bandwidth_min/model_step_1504800.ckpt \
#     --seed ${seed} \
#     --time-variant-bw

# exp_name=bo_T_s_bandwidth2
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}.json \
#     --pretrained-model-path test_bo_0421/bo_T_s_bandwidth_min/model_step_1504800.ckpt \
#     --seed ${seed} \
#     --time-variant-bw
# exp_name=bo_T_s_bandwidth_min
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#     --seed ${seed} \
#     --time-variant-bw

# exp_name=bo_T_s_delay_narrow
# CUDA_VISIBLE_DEVICES="" python train_rl.py \
#     --save-dir ${save_dir}/${exp_name} \
#     --exp-name ${exp_name}_seed_${seed} \
#     --tensorboard-log aurora_tensorboard \
#     --total-timesteps 5000000 \
#     --delta-scale 1 \
#     --randomization-range-file ../../config/train/udr_7_dims_0415/range0_${exp_name}0.json \
#     --pretrained-model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#     --seed ${seed} \
#     --time-variant-bw

save_dir=../../results_0430/udr_7_dims
# save_dir=test_mpi
exp_name=bo_delay7
#
#../../results_0426/udr_7_dims/bo_bw0/model_step_525600.ckpt \
# ../../results_0426/udr_7_dims/bo_bw1/model_step_1684800.ckpt \
for seed in 10 20 30 40 50; do
    CUDA_VISIBLE_DEVICES=""    python train_rl.py \
        --save-dir ${save_dir}/${exp_name}/seed_${seed} \
        --exp-name ${exp_name}_seed_${seed} \
        --tensorboard-log aurora_tensorboard \
        --total-timesteps 5000000 \
        --delta-scale 1 \
        --randomization-range-file ../../config/train/udr_7_dims_0430/${exp_name}.json \
        --pretrained-model-path ../../results_0430/udr_7_dims/bo_delay5/seed_50/model_step_604800.ckpt \
        --seed ${seed} \
        --time-variant-bw &
# ../../results_0430/udr_7_dims/range2/seed_50/model_step_1137600.ckpt \
# ../../results_0430/udr_7_dims/bo_delay0/seed_50/model_step_129600.ckpt \
# ../../results_0430/udr_7_dims/bo_delay1/seed_50/model_step_115200.ckpt \
# ../../results_0430/udr_7_dims/bo_delay2/seed_50/model_step_1562400.ckpt \
# ../../results_0430/udr_7_dims/bo_delay4/seed_50/model_step_122400.ckpt \
done
