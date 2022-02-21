
# save_root=/datamirror/zxxia/PCC-RL/results_1006/compare_udr_rule_based
#
#
# for seed in 10 20 30 40 50;do
#     for udr in small mid; do
#         model_path=/tank/zxxia/PCC-RL/results_0905/udr/udr_${udr}/seed_$seed/model_step_720000.ckpt
#         config_file=/tank/zxxia/PCC-RL/config/train/udr_7_dims_0827/udr_${udr}_seed_$seed.json
#         CUDA_VISIBLE_DEVICES="" python compare_udr_rule_based.py \
#             --heuristic bbr_old \
#             --model-path $model_path \
#             --config-file $config_file \
#             --seed $seed \
#             --save-dir $save_root/udr_${udr}/seed_$seed
#     done
#
#
#     model_path=/tank/zxxia/PCC-RL/results_0905/udr/udr_large/seed_$seed/model_step_720000.ckpt
#     config_file=/tank/zxxia/PCC-RL/config/train/udr_7_dims_0826/udr_large.json
#     CUDA_VISIBLE_DEVICES="" python compare_udr_rule_based.py \
#         --heuristic bbr_old \
#         --model-path $model_path \
#         --config-file $config_file \
#         --seed $seed \
#         --save-dir $save_root/udr_large/seed_$seed
# done

# save_root=/datamirror/zxxia/PCC-RL/results_1006/compare_udr_rule_based_same_test_traces
#
#
# for seed in 50;do
#     config_file=/tank/zxxia/PCC-RL/config/train/udr_7_dims_0827/udr_small_seed_10.json
#     for udr in  mid; do
#         model_path=/tank/zxxia/PCC-RL/results_0905/udr/udr_${udr}/seed_$seed/model_step_720000.ckpt
#         CUDA_VISIBLE_DEVICES="" python compare_udr_rule_based.py \
#             --heuristic bbr_old \
#             --model-path $model_path \
#             --config-file $config_file \
#             --seed $seed \
#             --save-dir $save_root/udr_${udr}/seed_$seed
#     done
#
#
#     model_path=/tank/zxxia/PCC-RL/results_0905/udr/udr_large/seed_$seed/model_step_720000.ckpt
#     CUDA_VISIBLE_DEVICES="" python compare_udr_rule_based.py \
#         --heuristic bbr_old \
#         --model-path $model_path \
#         --config-file $config_file \
#         --seed $seed \
#         --save-dir $save_root/udr_large/seed_$seed
# done



save_root=/datamirror/zxxia/PCC-RL/results_1006/compare_udr_rule_based_1023
save_root=/datamirror/zxxia/PCC-RL/results_1006/compare_udr_rule_based_1023_fix_large


for seed in 10 20 30 40 50;do
    # for udr in 1 2; do
    #     if [ $udr -eq 1 ]; then
    #         udr_name=small
    #     else
    #         udr_name=mid
    #     fi
    #     echo $udr $udr_name
    #     model_path=/tank/zxxia/PCC-RL/results_0928/udr${udr}/seed_$seed/model_step_720000.ckpt
    #     config_file=/tank/zxxia/PCC-RL/config/train/udr_7_dims_1023/udr_${udr_name}_seed_$seed.json
    #     CUDA_VISIBLE_DEVICES="" python compare_udr_rule_based.py \
    #         --heuristic cubic \
    #         --model-path $model_path \
    #         --config-file $config_file \
    #         --seed $seed \
    #         --save-dir $save_root/udr_${udr_name}/seed_$seed
    # done


    model_path=/tank/zxxia/PCC-RL/results_0928/udr3/seed_$seed/model_step_720000.ckpt
    config_file=/tank/zxxia/PCC-RL/config/train/udr_7_dims_0826/udr_large.json
    CUDA_VISIBLE_DEVICES="" python compare_udr_rule_based.py \
        --heuristic bbr_old \
        --model-path $model_path \
        --config-file $config_file \
        --seed $seed \
        --save-dir $save_root/udr_large/seed_$seed
done






# save_root=/datamirror/zxxia/PCC-RL/results_1006/compare_udr_rule_based_1023_same_test_traces
# config_file=/tank/zxxia/PCC-RL/config/train/udr_7_dims_1023/udr_small_seed_10.json
# for seed in 10;do
#     for udr in 1 2; do
#         if [ $udr -eq 1 ]; then
#             udr_name=small
#         else
#             udr_name=mid
#         fi
#         echo $udr $udr_name
#         model_path=/tank/zxxia/PCC-RL/results_0928/udr${udr}/seed_$seed/model_step_720000.ckpt
#         # config_file=/tank/zxxia/PCC-RL/config/train/udr_7_dims_1023/udr_${udr_name}_seed_$seed.json
#         CUDA_VISIBLE_DEVICES="" python compare_udr_rule_based.py \
#             --heuristic bbr_old \
#             --model-path $model_path \
#             --config-file $config_file \
#             --seed $seed \
#             --save-dir $save_root/udr_${udr_name}/seed_$seed
#     done
#
#
#     model_path=/tank/zxxia/PCC-RL/results_0928/udr3/seed_$seed/model_step_720000.ckpt
#     CUDA_VISIBLE_DEVICES="" python compare_udr_rule_based.py \
#         --heuristic bbr_old \
#         --model-path $model_path \
#         --config-file $config_file \
#         --seed $seed \
#         --save-dir $save_root/udr_large/seed_$seed
# done
#
