
#!/bin/bash

set -e

get_latest_model() {
    model_name=$(basename $(ls -t $1/model_step_*.ckpt.meta | head -n 1) .meta)
    echo $1/${model_name}
}

# for dimension in duration T_s_bandwidth T_s_delay bandwidth delay loss queue ; do
# for dimension in  T_s_delay  ; do
#         # test_bo_0421/bo_T_s_delay/model_step_756000.ckpt \
#     save_dir=fig8_udr_7_dims_2mbps_0421/bo_${dimension}
#     CUDA_VISIBLE_DEVICES="" python test_udr_vary_bw.py \
#         --model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#         test_bo_0421/bo_${dimension}_narrow/model_step_460800.ckpt \
#         --save-dir ../../results_0415/${save_dir} \
#         --dimension ${dimension} \
#         --config-file ../../config/test/fig8_7_dims_2mbps/rand_${dimension}.json \
#         --delta-scale 1 \
#         --train-config-dir ../../config/train/udr_7_dims_0415 \
#         --trace-dir bw_vary_traces_2mbps/rand_${dimension} \
#         --n-models 1 &
# done
#
#
#
for dimension in  T_s_bandwidth  ; do
        # test_bo_0421/bo_T_s_bandwidth/model_step_1749600.ckpt \
        # test_bo_0421/bo_T_s_bandwidth_narrow/model_step_1252800.ckpt \
    save_dir=fig8_udr_7_dims_2mbps_0421/bo_${dimension}
    CUDA_VISIBLE_DEVICES="" python test_udr_vary_bw.py \
        --model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
        test_bo_0421/bo_T_s_bandwidth_min/model_step_1504800.ckpt \
        test_bo_0421/bo_T_s_bandwidth1/model_step_266400.ckpt \
        test_bo_0421/bo_T_s_bandwidth2/model_step_381600.ckpt \
        --save-dir ../../results_0415/${save_dir} \
        --dimension ${dimension} \
        --config-file ../../config/test/fig8_7_dims_2mbps/rand_${dimension}.json \
        --delta-scale 1 \
        --train-config-dir ../../config/train/udr_7_dims_0415 \
        --trace-dir bw_vary_traces_2mbps_new/rand_${dimension} \
        --n-models 1
done



# for dimension in  bandwidth  ; do
#         # test_bo_0421/bo_bandwidth_10_12_single_env_const/model_step_612000.ckpt \
#         # test_bo_0421/bo_bandwidth_11_12_single_env/model_step_1432800.ckpt \
#         # test_bo_0421/bo_bandwidth_10_12_single_env_linear_0.002_24/model_step_612000.ckpt \
#     save_dir=fig8_udr_7_dims_2mbps_0421/bo_${dimension}
#     CUDA_VISIBLE_DEVICES="" python test_udr_vary_bw.py \
#         --model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#         test_bo_0421/bo_bandwidth_10_12_single_env/model_step_705600.ckpt \
#         --save-dir ../../results_0415/${save_dir} \
#         --dimension ${dimension} \
#         --config-file ../../config/test/fig8_7_dims_2mbps/rand_${dimension}.json \
#         --delta-scale 1 \
#         --train-config-dir ../../config/train/udr_7_dims_0415 \
#         --trace-dir bw_vary_traces_2mbps/rand_${dimension} \
#         --n-models 1
#         # test_bo_0421/bo_bandwidth_12_only/model_step_2390400.ckpt \
#         # test_bo_0421/bo_bandwidth_single_env/model_step_3312000.ckpt \
#         # test_bo_0421/bo_bandwidth_single_env_linear/model_step_2649600.ckpt \
# done


# for dimension in  delay  ; do
#     save_dir=fig8_udr_7_dims_2mbps_0421/bo_${dimension}
#     CUDA_VISIBLE_DEVICES="" python test_udr_vary_bw.py \
#         --model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt \
#         test_bo_0421/bo_${dimension}/model_step_1238400.ckpt \
#         --save-dir ../../results_0415/${save_dir} \
#         --dimension ${dimension} \
#         --config-file ../../config/test/fig8_7_dims_2mbps/rand_${dimension}.json \
#         --delta-scale 1 \
#         --train-config-dir ../../config/train/udr_7_dims_0415 \
#         --trace-dir bw_vary_traces_2mbps/rand_${dimension} \
#         --n-models 1
# done

