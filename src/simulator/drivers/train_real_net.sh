
SAVE_DIR=../../results_0326/model_for_real_net
SEED=42
CUDA_VISIBLE_DEVICES="" python train_rl.py \
    --save-dir ${SAVE_DIR}/tmp \
    --exp-name tmp \
    --duration 30 \
    --tensorboard-log aurora_tensorboard \
    --total-timesteps 5000000 \
    --delta-scale 1 \
    --randomization-range-file ../../config/train/real_net/home_wifi.json \
    --pretrained-model-path ../../results_0326/udr_4_dims_correct_recv_rate/range0/model_step_1504800.ckpt  \
    --seed ${SEED}
    # --save-dir ../../results_0326/${SAVE_DIR}/range0 \
