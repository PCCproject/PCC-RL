#!/bin/bash
# python train_rl.py --save-dir ../../results/bo_test/range1_bo_queue_0_correct  \
#     --exp-name range1_bo_queue_0 \
#     --randomization-range-file workspace/config/range1_bo_queue_0.json \
#     --pretrained-model-path ../../results/rand_4_dims_2mbps_default_sim/range_1/model_step_957600.ckpt
# python train_rl.py --save-dir ../../results/bo_test/range1_bo_queue_0_correct  \
#     --exp-name range1_bo_queue_0 \
#     --randomization-range-file workspace/config/range1_bo_queue_0.json \
#     --pretrained-model-path ../../results/rand_4_dims_2mbps_default_sim/range_1/model_step_957600.ckpt
# python train_rl.py --save-dir ../../results/bo_test/range1_bo_bandwidth_0  \
#     --exp-name range1_bo_bandwidth_0 \
#     --randomization-range-file workspace/config/range1_bo_bandwidth_0.json \
#     --pretrained-model-path ../../results/rand_4_dims_2mbps_default_sim/range_1/model_step_957600.ckpt
# python train_rl.py --save-dir ../../results/bo_test/range1_bo_delay_0  \
#     --total-timesteps 5000000 \
#     --exp-name range1_bo_delay_0 \
#     --randomization-range-file workspace/config/range1_bo_delay_0.json \
#     --pretrained-model-path ../../results/rand_4_dims_2mbps_default_sim/range_1/model_step_957600.ckpt
# python train_rl.py --save-dir ../../results/bo_test/range0_bo_bandwidth_0  \
#     --total-timesteps 5000000 \
#     --exp-name range0_bo_delay_0 \
#     --randomization-range-file workspace/config/range0_bo_delay_0.json \
#     --pretrained-model-path ../../results/rand_4_dims_2mbps_default_sim/range_0/model_step_223200.ckpt

# python train_rl.py --save-dir ../../results/bo_test/range0_bo_delay_1  \
#     --total-timesteps 5000000 \
#     --exp-name range0_bo_delay_1 \
#     --randomization-range-file workspace/config/range0_bo_delay_1.json \
#     --pretrained-model-path ../../results/rand_4_dims_2mbps_default_sim/range_0/model_step_223200.ckpt


# CUDA_VISIBLE_DEVICES="" python train_rl.py --save-dir ../../results/bo_test/range0_bo_delay_large_batch  \
#     --total-timesteps 5000000 \
#     --exp-name range0_bo_delay_large_batch \
#     --randomization-range-file workspace/config/range0_bo_delay_1.json \
#     --pretrained-model-path ../../results/rand_4_dims_2mbps_default_sim/range_0/model_step_223200.ckpt
# CUDA_VISIBLE_DEVICES="" python train_rl.py --save-dir ../../results/bo_test/range0_bo_delay_start_sending_rate_10  \
#     --total-timesteps 5000000 \
#     --exp-name range0_bo_delay_start_sending_rate_10 \
#     --randomization-range-file workspace/config/range0_bo_delay_1.json \
#     --pretrained-model-path ../../results/udr_4_dim/range0/model_step_1000800.ckpt
# CUDA_VISIBLE_DEVICES="" python train_rl.py --save-dir ../../results/bo_test/range0_bo_delay_start_sending_rate_10_scale_0.08  \
#     --total-timesteps 5000000 \
#     --exp-name range0_bo_delay_start_sending_rate_10_0.08 \
#     --randomization-range-file workspace/config/range0_bo_delay_1.json \
#     --pretrained-model-path ../../results/udr_4_dim/range0/model_step_1000800.ckpt
# CUDA_VISIBLE_DEVICES="" python train_rl.py --save-dir ../../results/bo_test/range0_bo_delay_start_sending_rate_10_scale_0.1  \
#     --total-timesteps 5000000 \
#     --exp-name range0_bo_delay_start_sending_rate_10_0.1 \
#     --randomization-range-file workspace/config/range0_bo_delay_1.json \
#     --pretrained-model-path ../../results/udr_4_dim/range0/model_step_1000800.ckpt
# CUDA_VISIBLE_DEVICES="" python train_rl.py --save-dir ../../results/bo_test/range0_bo_bandwidth_start_sending_rate_10_scale_0.05  \
#     --total-timesteps 5000000 \
#     --exp-name range0_bo_bandwidth_start_sending_rate_10_0.05 \
#     --randomization-range-file workspace/config/range0_bo_bandwidth_0.json \
#     --pretrained-model-path ../../results/udr_4_dim/range0/model_step_1000800.ckpt
# CUDA_VISIBLE_DEVICES="" python train_rl.py --save-dir ../../results/bo_test/range0_bo_queue_start_sending_rate_10_scale_0.05  \
#     --total-timesteps 5000000 \
#     --exp-name range0_bo_queue_start_sending_rate_10_0.05 \
#     --randomization-range-file workspace/config/range0_bo_queue_0.json \
#     --pretrained-model-path ../../results/udr_4_dim/range0/model_step_1000800.ckpt
CUDA_VISIBLE_DEVICES="" python train_rl.py --save-dir ../../results/bo_test/range0_bo_bandwidth_start_sending_rate_10_scale_0.1  \
    --total-timesteps 5000000 \
    --exp-name range0_bo_bandwidth_start_sending_rate_10_0.1 \
    --randomization-range-file workspace/config/range0_bo_bandwidth_0.json \
    --pretrained-model-path ../../results/udr_4_dim/range0/model_step_1000800.ckpt
