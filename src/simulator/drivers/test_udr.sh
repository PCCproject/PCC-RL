#!/bin/bash

set -e

# python test_udr_new.py \
#     --model-path ../../results/udr_4_dim_new/range0/model_step_4284000.ckpt  \
#     ../../results/udr_4_dim_new/range1/model_step_3398400.ckpt \
#     ../../results/udr_4_dim_new/range2/model_step_3384000.ckpt \
#     ../../results/udr_4_dim_new/range0_new/model_step_1598400.ckpt \
#     --save-dir fig8_tmp \
#     --dimension bandwidth \
#     --config-file ../../config/test/fig8/rand_bandwidth.json \
#     --duration 10


# python test_udr_new.py \
#     --model-path ../../results/udr_4_dim_2s/range0/model_step_1072800.ckpt  \
#     ../../results/udr_4_dim_2s/range1/model_step_1274400.ckpt \
#     ../../results/udr_4_dim_2s/range2/model_step_1879200.ckpt \
#     --save-dir fig8_2s \
#     --dimension bandwidth \
#     --config-file ../../config/test/fig8/rand_bandwidth.json \
#     --duration 2

# python test_udr_new.py \
#     --model-path ../../results/udr_4_dim_fix_reset/range0/model_step_2066400.ckpt  \
#     ../../results/udr_4_dim_fix_reset/range1/model_step_2275200.ckpt \
#     ../../results/udr_4_dim_fix_reset/range2/model_step_2347200.ckpt \
#     --save-dir fig8_fix_reset \
#     --dimension bandwidth \
#     --config-file ../../config/test/fig8/rand_bandwidth.json \
#     --duration 10 \
#     --delta-scale 0.05

# python test_udr_new.py \
#     --model-path \
#     ../../results/udr_4_dim_scale_1/range0/model_step_1504800.ckpt \
#     ../../results/udr_4_dim_scale_1/range1/model_step_2865600.ckpt  \
#     ../../results/udr_4_dim_scale_1/range2/model_step_3384000.ckpt \
#     ../../results/udr_4_dim_scale_1/range1_bo/model_step_2952000.ckpt \
#     --save-dir fig8_scale_1 \
#     --dimension bandwidth \
#     --config-file ../../config/test/fig8/rand_bandwidth.json \
#     --duration 10 \
#     --delta-scale 1

# dimension=loss
# python test_udr_new.py \
#     --plot-only \
#     --model-path \
#     ../../results/udr_4_dim_scale_1/range0/model_step_1504800.ckpt \
#     ../../results/udr_4_dim_scale_1/range1/model_step_2865600.ckpt  \
#     ../../results/udr_4_dim_scale_1/range2/model_step_3384000.ckpt \
#     --save-dir fig8_scale_1 \
#     --dimension ${dimension} \
#     --config-file ../../config/test/fig8/rand_${dimension}.json \
#     --duration 10 \
#     --delta-scale 1
    # ../../results/udr_4_dim_scale_1/range1_bo_${dimension}/model_step_6940800.ckpt \


# dimension=loss
# python test_udr_new.py \
#     --model-path \
#     ../../results/udr_4_dim_scale_1_new/range0/model_step_1504800.ckpt \
#     ../../results/udr_4_dim_scale_1_new/range1/model_step_3391200.ckpt  \
#     ../../results/udr_4_dim_scale_1_new/range2/model_step_1555200.ckpt \
#     --save-dir fig8_scale_1_new \
#     --dimension ${dimension} \
#     --config-file ../../config/test/fig8/rand_${dimension}.json \
#     --duration 10 \
#     --delta-scale 1

# python test_udr_new.py \
#     --model-path \
#     ../../results/udr_4_dim_scale_1/range0/model_step_1504800.ckpt \
#     ../../results/udr_4_dim_scale_1/range1/model_step_2865600.ckpt  \
#     ../../results/udr_4_dim_scale_1/range2/model_step_3384000.ckpt \
#     ../../results/udr_4_dim_scale_1/range1_bo/model_step_2952000.ckpt \
#     --save-dir fig8_scale_1 \
#     --dimension queue \
#     --config-file ../../config/test/fig8/rand_queue.json \
#     --duration 10 \
#     --delta-scale 1

# python test_udr_new.py \
#     --model-path \
#     ../../results/udr_4_dim_scale_0.01/range0/model_step_2937600.ckpt \
#     ../../results/udr_4_dim_scale_0.01/range1/model_step_4968000.ckpt  \
#     ../../results/udr_4_dim_scale_0.01/range2/model_step_1936800.ckpt \
#     --save-dir fig8_scale_0.01 \
#     --dimension bandwidth \
#     --config-file ../../config/test/fig8/rand_bandwidth.json \
#     --duration 10 \
#     --delta-scale 0.01


# dimension=bandwidth
# python test_udr_new.py \
#     --model-path \
#     ../../results/udr_4_dim_scale_1/range0/model_step_1504800.ckpt \
#     ../../results/udr_4_dim_scale_1/range1/model_step_2865600.ckpt  \
#     ../../results/udr_4_dim_scale_1/range2/model_step_3384000.ckpt \
#     ../../results/udr_4_dim_scale_1/range1_bo/model_step_2952000.ckpt \
#     ../../results/udr_4_dim_scale_1/range1_bo_bandwidth_28/model_step_3254400.ckpt  \
#     ../../results/udr_4_dim_scale_1/range1_bo_bandwidth_37/model_step_3902400.ckpt  \
#     ../../results/udr_4_dim_scale_1/range1_bo_bandwidth_46/model_step_3254400.ckpt  \
#     ../../results/udr_4_dim_scale_1/range1_bo_bandwidth_55/model_step_3254400.ckpt  \
#     --save-dir fig8_scale_1 \
#     --dimension ${dimension} \
#     --config-file ../../config/test/fig8/rand_${dimension}.json \
#     --duration 10 \
#     --delta-scale 1 \
#     --n-models 3

# dimension=delay
# python test_udr_new.py \
#     --model-path \
#     ../../results/udr_4_dim_scale_1/range0/model_step_1504800.ckpt \
#     ../../results/udr_4_dim_scale_1/range1/model_step_2865600.ckpt  \
#     ../../results/udr_4_dim_scale_1/range2/model_step_3384000.ckpt \
#     ../../results/udr_4_dim_scale_1/range1_bo_delay/model_step_3808800.ckpt  \
#     ../../results/udr_4_dim_scale_1/range1_bo_delay_55/model_step_3175200.ckpt  \
#     ../../results/udr_4_dim_scale_1_new/range1_bo_delay_64/model_step_3182400.ckpt  \
#     ../../results/udr_4_dim_scale_1_new/range1_bo_delay_73/model_step_3196800.ckpt  \
#     --save-dir fig8_scale_1 \
#     --dimension ${dimension} \
#     --config-file ../../config/test/fig8/rand_${dimension}.json \
#     --duration 10 \
#     --delta-scale 1 \
#     --n-models 3
# dimension=delay
# dimension=bandwidth
# python test_udr_new.py \
#     --model-path \
#     ../../results/udr_4_dim_scale_1/range0/model_step_1504800.ckpt \
#     ../../results/udr_4_dims/range1/model_step_5004000.ckpt  \
#     ../../results/udr_4_dims/range2/model_step_5004000.ckpt \
#     --save-dir fig8 \
#     --dimension ${dimension} \
#     --config-file ../../config/test/fig8/rand_${dimension}.json \
#     --duration 10 \
#     --delta-scale 1 \
#     --n-models 1

# dimension=delay
# dimension=bandwidth
# dimension=loss
# dimension=queue
# for dimension in delay bandwidth loss queue; do
# CUDA_VISIBLE_DEVICES="" python test_udr_new.py \
#     --model-path \
#     ../../results_0326/udr_4_dims_correct_recv_rate/range0/model_step_5004000.ckpt \
#     ../../results_0326/udr_4_dims_correct_recv_rate/range1/model_step_5004000.ckpt  \
#     ../../results_0326/udr_4_dims_correct_recv_rate/range2/model_step_4658400.ckpt \
#     --save-dir fig8_0326 \
#     --dimension ${dimension} \
#     --config-file ../../config/test/fig8/rand_${dimension}.json \
#     --duration 10 \
#     --delta-scale 1 \
#     --n-models 1
#     # ../../results_0326/udr_4_dims_correct_recv_rate/range0/model_step_1116000.ckpt \
#     # ../../results_0326/udr_4_dims_correct_recv_rate/range1/model_step_295200.ckpt  \
#     # ../../results_0326/udr_4_dims_correct_recv_rate/range2/model_step_288000.ckpt \
# done

get_latest_model() {
    model_name=$(basename $(ls -t $1/model_step_*.ckpt.meta | head -n 1) .meta)
    echo $1/${model_name}
    # return ${1}/${model_name}
}

for dimension in delay bandwidth loss queue; do
    # $(get_latest_model ../../results_0326/udr_4_dims_log_queue/range0)
CUDA_VISIBLE_DEVICES="" python test_udr_new.py \
    --model-path \
    ../../results_0326/udr_4_dims_log_queue/range0/model_step_2304000.ckpt \
    ../../results_0326/udr_4_dims_log_queue/range1/model_step_2304000.ckpt \
    ../../results_0326/udr_4_dims_log_queue/range2/model_step_2304000.ckpt \
    ../../results_0326/udr_4_dims_log_queue/range4/model_step_2304000.ckpt \
    ../../results_0326/udr_4_dims_log_queue/range6/model_step_2304000.ckpt \
    --save-dir fig8_0327 \
    --dimension ${dimension} \
    --config-file ../../config/test/fig8/rand_${dimension}.json \
    --duration 10 \
    --delta-scale 1 \
    --train-config-dir ../../config/train/udr_4_dims_0327 \
    --n-models 1
    # $(get_latest_model ../../results_0326/udr_4_dims_log_queue/range0) \
    # $(get_latest_model ../../results_0326/udr_4_dims_log_queue/range1) \
    # $(get_latest_model ../../results_0326/udr_4_dims_log_queue/range2) \
    # $(get_latest_model ../../results_0326/udr_4_dims_log_queue/range3) \
    # $(get_latest_model ../../results_0326/udr_4_dims_log_queue/range4) \
    # $(get_latest_model ../../results_0326/udr_4_dims_log_queue/range5) \
    # $(get_latest_model ../../results_0326/udr_4_dims_log_queue/range6) \
    # $(get_latest_model ../../results_0326/udr_4_dims_log_queue/range7) \

    # ../../results_0326/udr_4_dims_correct_recv_rate_vary_send_rate/range0/model_step_2916000.ckpt \
    # ../../results_0326/udr_4_dims_correct_recv_rate_vary_send_rate/range1/model_step_1080000.ckpt  \
    # ../../results_0326/udr_4_dims_correct_recv_rate_vary_send_rate/range2/model_step_1173600.ckpt \
    # ../../results_0326/udr_4_dims_correct_recv_rate/range0/model_step_1116000.ckpt \
    # ../../results_0326/udr_4_dims_correct_recv_rate/range1/model_step_295200.ckpt  \
    # ../../results_0326/udr_4_dims_correct_recv_rate/range2/model_step_288000.ckpt \
done


# #     # ../../results/udr_4_dim_scale_1/range1_bo/model_step_2952000.ckpt \
#     # ../../results/udr_4_dim_scale_1/range1_bo_delay_28/model_step_3038400.ckpt  \
#     # ../../results/udr_4_dim_scale_1/range1_bo_delay_37/model_step_3146400.ckpt  \
#     # ../../results/udr_4_dim_scale_1/range1_bo_delay_46/model_step_3628800.ckpt  \


# log_files=$(ls fig8_tmp/rand_bandwidth/*/cubic/cubic_test_log.csv)
# log_files=$(ls fig8_scale_1/rand_bandwidth/env_2.5_50_0_10/cubic/cubic_test_log.csv)
# log_files=$(ls fig8_scale_1/rand_bandwidth/env_2.5_50_0_10/range1/aurora_test_log.csv)
# log_files=$(ls fig8_scale_1/rand_bandwidth/env_2.5_50_0_10/range0/aurora_test_log.csv)
# log_files=$(ls fig8_tmp/rand_bandwidth/env_2.5_50_0_10/range0/aurora_test_log.csv)
# log_files=$(ls fig8_fix_reset/rand_bandwidth/env_2.5_50_0_10/range1/aurora_test_log.csv)
# log_files=$(ls fig8_fix_reset/rand_bandwidth/env_2.5_50_0_10/range0/aurora_test_log.csv)
# log_files=$(ls fig8_tmp/rand_bandwidth/env_2.5_50_0_10/range2/aurora_test_log.csv)
# log_files=$(ls fig8_tmp/rand_bandwidth/env_2.5_50_0_10/range0_new/aurora_test_log.csv)
# log_files=$(ls fig8_scale_1/rand_bandwidth/*/cubic/cubic_test_log.csv)
# log_files=$(ls fig8_scale_1/rand_bandwidth/*/range0/aurora_test_log.csv)
# log_files=$(ls fig8_scale_1/rand_bandwidth/*/range1/aurora_test_log.csv)
# log_files=$(ls fig8_scale_1/rand_bandwidth/*/range2/aurora_test_log.csv)
# log_files=$(ls fig8_scale_1/rand_bandwidth/env_2.5_50_0_10/range1_bo/aurora_test_log.csv)
# log_files=$(ls fig8_fix_reset/rand_bandwidth/*/cubic/cubic_test_log.csv)
# log_files=$(ls fig8_fix_reset/rand_bandwidth/*/range0/aurora_test_log.csv)
# log_files=$(ls fig8_fix_reset/rand_bandwidth/*/range1/aurora_test_log.csv)
# log_files=$(ls fig8_fix_reset/rand_bandwidth/*/range2/aurora_test_log.csv)

# log_files=$(ls fig8_scale_0.01/rand_bandwidth/*/cubic/cubic_test_log.csv)
# log_files=$(ls fig8_scale_0.01/rand_bandwidth/*/range0/aurora_test_log.csv)
# log_files=$(ls fig8_scale_0.01/rand_bandwidth/*/range1/aurora_test_log.csv)
# log_files=$(ls fig8_scale_0.01/rand_bandwidth/*/range2/aurora_test_log.csv)
# for log_file in ${log_files}; do
#     echo ${log_file}
#     save_dir=$(dirname $log_file)
#     python ../plot_scripts/plot_time_series.py --log-file ${log_file} --save-dir ${save_dir}
# done
