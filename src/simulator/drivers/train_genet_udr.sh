
save_root=/datamirror/zxxia/PCC-RL/results_1006
# exp_name=genet_random_real
exp_name=genet_udr
for seed in 10 20 30; do
    for prob in 0.1; do
    # for prob in 0.04 0.06 0.05 0.2; do
        save_dir=${save_root}/${exp_name}/${prob}/seed_${seed}
        CUDA_VISIBLE_DEVICES="" python genet_udr.py \
            --seed ${seed} \
            --heuristic bbr_old \
            --save-dir ${save_dir} \
            --config-file ../../config/train/udr_7_dims_0826/udr_large.json \
            --bo-rounds 15 \
            --validation \
            --type bo \
            --model-select latest \
            --real-trace-prob $prob \
            --model-path ../../results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt &
    done
done
