#!/bin/bash

set -e
save_root=/datamirror/zxxia/PCC-RL/results_1006/train_with_difficulty_metrics

# exp_name=train_with_difficulty_metrics/metric2
# exp_name=train_with_difficulty_metrics/metric2_new
# exp_name=train_with_difficulty_metrics/metric2_new_debug
# exp_name=train_with_difficulty_metrics/metric2_diff0
# exp_name=train_with_difficulty_metrics/metric2_new_noise
# exp_name=train_with_difficulty_metrics/metric2_cubic
config_file=../../config/train/udr_7_dims_0826/udr_large.json
exp_name=metric3_new
for seed in 10 20 30; do
    save_dir=${save_root}/${exp_name}/seed_${seed}
    echo ${exp_name}
    echo ${save_dir}



    CUDA_VISIBLE_DEVICES="" python genet_improved.py \
        --heuristic optimal \
        --save-dir ${save_dir} \
        --bo-rounds 15 \
        --seed ${seed} \
        --validation \
        --config-file  ${config_file} &
        # --model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt &

done
