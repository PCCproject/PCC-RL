#!/bin/bash

set -e

get_latest_model() {
    model_name=$(basename $(ls -t $1/model_step_*.ckpt.meta | head -n 1) .meta)
    echo $1/${model_name}
}

# for dimension in delay bandwidth loss queue; do
# for dimension in duration prob_stay timestep; do
#     # $(get_latest_model ../../results_0326/udr_4_dims_log_queue/range0)
#     CUDA_VISIBLE_DEVICES="" python test_udr_vary_bw.py \
#         --model-path \
#         $(get_latest_model ../../results_0326/udr_7_dims/range1) \
#         $(get_latest_model ../../results_0326/udr_7_dims/range2) \
#         $(get_latest_model ../../results_0326/udr_7_dims/range3) \
#         --save-dir fig8_udr_7_dims \
#         --dimension ${dimension} \
#         --config-file ../../config/test/fig8_7_dims/rand_${dimension}.json \
#         --delta-scale 1 \
#         --train-config-dir ../../config/train/udr_7_dims \
#         --trace-dir rand_${dimension} \
#         --n-models 1
# done
# for dimension in duration prob_stay timestep; do
#     # $(get_latest_model ../../results_0326/udr_4_dims_log_queue/range0)
#     traces=$(ls fig8_udr_7_dims/rand_${dimension}/)
#     for trace in ${traces}; do
#     #   for range_id in 1 2 3; do
#     #         model_name=$(ls -t fig8_udr_7_dims/rand_${dimension}/${trace}/range${range_id} | head -n 1)
#     #         python ../plot_scripts/plot_time_series.py \
#     #                 --log-file fig8_udr_7_dims/rand_${dimension}/${trace}/range${range_id}/${model_name}/aurora_test_log.csv \
#     #                 --save-dir fig8_udr_7_dims/rand_${dimension}/${trace}/range${range_id}/${model_name}
#     #     done
#         python ../plot_scripts/plot_time_series.py \
#                 --log-file fig8_udr_7_dims/rand_${dimension}/${trace}/cubic/cubic_test_log.csv \
#                 --save-dir fig8_udr_7_dims/rand_${dimension}/${trace}/cubic
#     done
# done


# duration T_s_bandwidth T_s_delay bandwidth delay
# for dimension in delay bandwidth loss queue; do
        # ../../results_0415/udr_7_dims/range0/model_step_1036800.ckpt \
        # ../../results_0415/udr_7_dims/range0/model_step_2865600.ckpt \
        # ../../results_0415/udr_7_dims/range1/model_step_489600.ckpt \
        # ../../results_0415/udr_7_dims/range2/model_step_151200.ckpt \
        # ../../results_0415/udr_7_dims/range0_vary_bw/model_step_1576800.ckpt \
        # ../../results_0415/udr_7_dims/range0/model_step_2865600.ckpt \
# for dimension in loss queue; do
#
# for dimension in bandwidth; do
# for dimension in duration T_s_bandwidth T_s_delay bandwidth delay loss queue ; do
# trace_dir=bw_vary_traces_2mbps/rand_${dimension}
# trace_dir=../../data/synthetic_traces
trace_dir=../../data/synthetic_traces_30s
# save_dir=../../results_0423/fig8_udr_7_dims
# save_dir=../../results_0423/fig8_udr_7_dims_tmp
save_dir=../../results_0430/fig8_udr_7_dims
#delay_noise d_bw T_s d_delay queue loss duration bandwidth
 # queue loss duration bandwidth delay d_bw T_s delay_noise
    for dimension in bandwidth; do
    # for dimension in d_bw; do
    # $(get_latest_model ../../results_0326/udr_4_dims_log_queue/range0)
        # test_bo/delay/model_step_3218400.ckpt \
        # ../../results_0415/udr_7_dims/range0_vary_bw_cont/model_step_1850400.ckpt \
        # ../../results_0415/udr_7_dims/range0_vary_bw_6s_cont/model_step_1670400.ckpt \

        # ../../results_0415/udr_7_dims/range1/model_step_4262400.ckpt \
        # ../../results_0415/udr_7_dims/range2/model_step_1440000.ckpt \

        # bo_loss0 ../../results_0423/udr_7_dims/bo_loss0/model_step_295200.ckpt \
        # bo_loss1 ../../results_0423/udr_7_dims/bo_loss1/model_step_460800.ckpt \
        # bo_bw0 ../../results_0423/udr_7_dims/bo_bw0/model_step_504000.ckpt \
        # ../../results_0423/udr_7_dims/bo_bw1/model_step_374400.ckpt \

        # bo_bw0 ../../results_0423/udr_7_dims/bo_bw0/model_step_424800.ckpt \
        # bo_bw1 ../../results_0423/udr_7_dims/bo_bw1/model_step_273600.ckpt \
        # bo_duration0 ../../results_0423/udr_7_dims/bo_duration0/model_step_201600.ckpt \
        # ../../results_0423/udr_7_dims/bo_bw2/model_step_165600.ckpt \
        # ../../results_0423/udr_7_dims/bo_d_bw0/model_step_108000.ckpt \
            # ../../results_0423/udr_7_dims/range0/model_step_1447200.ckpt \

        # ../../results_0426/udr_7_dims/bo_bw0/model_step_525600.ckpt \
        # --model-path ../../results_0426/udr_7_dims/range0/model_step_669600.ckpt \
        # ../../results_0426/udr_7_dims/bo_d_bw0/model_step_72000.ckpt \

        #     ../../results_0426/udr_7_dims/range0_before_change_feature/model_step_1202400.ckpt \
        # ../../results_0426/udr_7_dims/bo_bw1/model_step_1684800.ckpt \
        # ../../results_0426/udr_7_dims/bo_delay0/model_step_1468800.ckpt \
        # ../../results_0426/udr_7_dims/bo_delay1/model_step_777600.ckpt \

# ../../results_0430/udr_7_dims/bo_delay0/seed_50/model_step_129600.ckpt \
# ../../results_0430/udr_7_dims/bo_delay1/seed_50/model_step_115200.ckpt \

# ../../results_0430/udr_7_dims/bo_delay1/seed_50/model_step_115200.ckpt \
#     ../../results_0430/udr_7_dims/bo_delay2/seed_10/model_step_576000.ckpt \
#     ../../results_0430/udr_7_dims/bo_delay3/seed_10/model_step_237600.ckpt \
# ../../results_0430/udr_7_dims/bo_delay5/seed_50/model_step_604800.ckpt \
            # ../../results_0430/udr_7_dims/range2/seed_50/model_step_1137600.ckpt \
            # ../../results_0426/udr_7_dims/range0_before_change_feature/model_step_1202400.ckpt \
    CUDA_VISIBLE_DEVICES="" python test_udr_vary_bw_new.py \
        --save-dir ${save_dir} \
        --model-path ../../results_0503/udr_7_dims/udr_small/seed_10/model_step_180000.ckpt \
../../results_0503/udr_7_dims/udr_mid/seed_50/model_step_360000.ckpt \
            ../../results_0503/udr_7_dims/udr_large/seed_50/model_step_396000.ckpt \
        --dimension ${dimension} \
        --config-file ../../config/test/fig8_7_dims_2mbps/rand_${dimension}.json \
        --delta-scale 1 \
        --train-config-dir ../../config/train/udr_7_dims_0423 \
        --trace-dir ${trace_dir}/rand_${dimension} \
        --n-models 1
            #\
        # --plot-only
        # $(get_latest_model ../../results_0415/udr_7_dims/range0) \
        # $(get_latest_model ../../results_0415/udr_7_dims/range1) \
        # $(get_latest_model ../../results_0415/udr_7_dims/range2) \
done

# for dimension in duration prob_stay timestep; do
#     # $(get_latest_model ../../results_0326/udr_4_dims_log_queue/range0)
#     traces=$(ls fig8_udr_7_dims/rand_${dimension}/)
#     for trace in ${traces}; do
#     #   for range_id in 1 2 3; do
#     #         model_name=$(ls -t fig8_udr_7_dims/rand_${dimension}/${trace}/range${range_id} | head -n 1)
#     #         python ../plot_scripts/plot_time_series.py \
#     #                 --log-file fig8_udr_7_dims/rand_${dimension}/${trace}/range${range_id}/${model_name}/aurora_test_log.csv \
#     #                 --save-dir fig8_udr_7_dims/rand_${dimension}/${trace}/range${range_id}/${model_name}
#     #     done
#         python ../plot_scripts/plot_time_series.py \
#                 --log-file fig8_udr_7_dims/rand_${dimension}/${trace}/cubic/cubic_test_log.csv \
#                 --save-dir fig8_udr_7_dims/rand_${dimension}/${trace}/cubic
#     done
# done
