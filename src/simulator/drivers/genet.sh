set -e
SAVE_DIR=../../results_0826/genet_cubic_exp_2
CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python genet.py \
    --heuristic cubic \
    --save-dir ${SAVE_DIR} \
    --config-file ../../config/train/udr_7_dims_0826/udr_large.json \
    --bo-rounds 200 \
    --model-path ../../results_0826/udr_7/udr_large/seed_10/model_step_21600.ckpt
    # --model-path ../../results_0826/udr_6/udr_large/seed_20/model_step_21600.ckpt
    # --model-path ../../results_0826/udr_3/udr_large/seed_20/model_step_151200.ckpt

SAVE_DIR=../../results_0826/genet_bbr_exp_2
CUDA_VISIBLE_DEVICES="" mpiexec -np 2 python genet.py \
    --heuristic bbr \
    --save-dir ${SAVE_DIR}/ \
    --config-file ../../config/train/udr_7_dims_0826/udr_large.json \
    --bo-rounds 200 \
    --model-path ../../results_0826/udr_7/udr_large/seed_10/model_step_36000.ckpt
    # --model-path ../../results_0826/udr_6/udr_large/seed_20/model_step_21600.ckpt
    # --model-path ../../results_0826/udr_3/udr_large/seed_20/model_step_21600.ckpt
