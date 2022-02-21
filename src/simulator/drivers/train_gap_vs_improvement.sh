SAVE_DIR='/datamirror/zxxia/PCC-RL/results_1006/gap_vs_improvement_pretrained'
SAVE_DIR='/datamirror/zxxia/PCC-RL/results_1006/gap_vs_improvement'
seed=10
exp_name=train


# pretrained_model=../../results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt
pretrained_model=/tank/zxxia/PCC-RL/results_0928/genet_no_reward_scale/genet_bbr_old/seed_10/bo_0/model_step_64800.ckpt
END=110
for i in $(seq 100 10 $END); do
    pids=""
    for j in $(seq 0 9); do
        k=$((i + j))
        echo ${k}
        # config_file=../../config/gap_vs_improvement_pretrained/config_${k}.json
        config_file=../../config/gap_vs_improvement/config_${k}.json
        CUDA_VISIBLE_DEVICES="" mpiexec -np 4 python train_rl.py \
            --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed}/config_${k} \
            --total-timesteps 72000 \
            --randomization-range-file $config_file  \
            --seed ${seed} \
            --validation \
            --pretrained-model-path $pretrained_model &
        pids="$pids $!"
    done
    wait $pids
done

# pids=""
# for i in 30 31 32 33 34 35 36 37 38 39; do
#     CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
#         --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed}/config_${i} \
#         --total-timesteps 72000 \
#         --randomization-range-file ../../config/gap_vs_improvement/config_${i}.json \
#         --seed ${seed} \
#         --validation \
#         --pretrained-model-path /tank/zxxia/PCC-RL/results_0928/genet_no_reward_scale/genet_bbr_old/seed_10/bo_0/model_step_64800.ckpt  &
#     pids="$pids $!"
# done
#
# wait $pids
#
# pids=""
# for i in 40 41 42 43 44 45 46 47 48 49; do
#     CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
#         --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed}/config_${i} \
#         --total-timesteps 72000 \
#         --randomization-range-file ../../config/gap_vs_improvement/config_${i}.json \
#         --seed ${seed} \
#         --validation \
#         --pretrained-model-path /tank/zxxia/PCC-RL/results_0928/genet_no_reward_scale/genet_bbr_old/seed_10/bo_0/model_step_64800.ckpt  &
#     pids="$pids $!"
# done
#
#
# pids=""
# for i in 50 51 52 53 54 55 56 57 58 59; do
#     CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
#         --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed}/config_${i} \
#         --total-timesteps 72000 \
#         --randomization-range-file ../../config/gap_vs_improvement/config_${i}.json \
#         --seed ${seed} \
#         --validation \
#         --pretrained-model-path /tank/zxxia/PCC-RL/results_0928/genet_no_reward_scale/genet_bbr_old/seed_10/bo_0/model_step_64800.ckpt  &
#     pids="$pids $!"
# done
# wait $pids
#
#
# pids=""
# for i in 60 61 62 63 64 65 66 67 68 69; do
#     CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
#         --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed}/config_${i} \
#         --total-timesteps 72000 \
#         --randomization-range-file ../../config/gap_vs_improvement/config_${i}.json \
#         --seed ${seed} \
#         --validation \
#         --pretrained-model-path /tank/zxxia/PCC-RL/results_0928/genet_no_reward_scale/genet_bbr_old/seed_10/bo_0/model_step_64800.ckpt  &
# done
# wait $pids
#
#
# pids=""
# for i in 70 71 72 73 74 75 76 77 78 79; do
#     CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
#         --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed}/config_${i} \
#         --total-timesteps 72000 \
#         --randomization-range-file ../../config/gap_vs_improvement/config_${i}.json \
#         --seed ${seed} \
#         --validation \
#         --pretrained-model-path /tank/zxxia/PCC-RL/results_0928/genet_no_reward_scale/genet_bbr_old/seed_10/bo_0/model_step_64800.ckpt  &
#     pids="$pids $!"
# done
# wait $pids
#
#
# pids=""
# for i in 80 81 82 83 84 85 86 87 88 89; do
#     CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
#         --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed}/config_${i} \
#         --total-timesteps 72000 \
#         --randomization-range-file ../../config/gap_vs_improvement/config_${i}.json \
#         --seed ${seed} \
#         --validation \
#         --pretrained-model-path /tank/zxxia/PCC-RL/results_0928/genet_no_reward_scale/genet_bbr_old/seed_10/bo_0/model_step_64800.ckpt  &
#     pids="$pids $!"
# done
# wait $pids
#
# for i in 90 91 92 93 94 95 96 97 98 99; do
#     CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python train_rl.py \
#         --save-dir ${SAVE_DIR}/${exp_name}/seed_${seed}/config_${i} \
#         --total-timesteps 72000 \
#         --randomization-range-file ../../config/gap_vs_improvement/config_${i}.json \
#         --seed ${seed} \
#         --validation \
#         --pretrained-model-path /tank/zxxia/PCC-RL/results_0928/genet_no_reward_scale/genet_bbr_old/seed_10/bo_0/model_step_64800.ckpt  &
# done
