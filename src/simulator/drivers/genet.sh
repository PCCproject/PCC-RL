set -e
SAVE_DIR=../../results_0911/genet_cubic_noise_udr_start
CUDA_VISIBLE_DEVICES="" mpiexec -np 8 python genet.py \
    --heuristic cubic \
    --save-dir ${SAVE_DIR} \
    --config-file ../../config/train/udr_7_dims_0911/udr_large.json \
    --bo-rounds 200 \
    --model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_21600.ckpt &
    # --model-path ../../results_0826/udr_7/udr_large/seed_10/model_step_21600.ckpt &
    # --model-path ../../results_0826/udr_6/udr_large/seed_20/model_step_21600.ckpt
    # --model-path ../../results_0826/udr_3/udr_large/seed_20/model_step_151200.ckpt

# SAVE_DIR=../../results_0826/genet_bbr_delete
SAVE_DIR=../../results_0911/genet_bbr_noise_udr_start
CUDA_VISIBLE_DEVICES="" mpiexec -np 8 python genet.py \
    --heuristic bbr \
    --save-dir ${SAVE_DIR}/ \
    --config-file ../../config/train/udr_7_dims_0911/udr_large.json \
    --bo-rounds 200 \
    --model-path /tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_21600.ckpt &
# ../../results_0826/udr_7/udr_large/seed_10/model_step_21600.ckpt &
    # --model-path ../../results_0826/udr_6/udr_large/seed_20/model_step_21600.ckpt
    # --model-path ../../results_0826/udr_3/udr_large/seed_20/model_step_21600.ckpt
