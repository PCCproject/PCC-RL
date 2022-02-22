# for seed in 10 20 30 40 50; do
for seed in 60 70 80 90 100 110 120 130 140 150; do
    for bo in 0 1 2 3 4 5 6; do
        for model_seed in 10;do
            # SAVE_DIR=../../results_0928/verify_useless_config_random/genet_bbr_old/seed_${seed}
            # SAVE_DIR=../../results_0928/no_reward_scale/genet_bbr_old/seed_${seed}
            # SAVE_DIR=../../results_0928/manual_verify_useless_config/genet_bbr_old/seed_${seed}
            exp_name=run_multiple_seed_bo
            SAVE_DIR=/datamirror/zxxia/results_1006/${exp_name}/model_seed_${model_seed}/bo_${bo}/seed_${seed}
            CUDA_VISIBLE_DEVICES="" python genet_improved.py \
                --seed ${seed} \
                --heuristic bbr_old \
                --save-dir ${SAVE_DIR} \
                --config-file ../../config/train/udr_7_dims_0826/udr_large.json \
                --bo-rounds 2 \
                --validation \
                --type bo \
                --model-path /tank/zxxia/PCC-RL/results_0928/genet_no_reward_scale/genet_bbr_old/seed_$model_seed/bo_$bo/model_step_64800.ckpt
        done
    done
done

