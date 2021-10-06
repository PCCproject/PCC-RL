#!/bin/bash

# python ../plot_scripts/plot_validation.py --log-file \
#     test_bo_0421/bo_bandwidth/validation_log.csv \
#     test_bo_0421/bo_bandwidth_10_12_single_env/validation_log.csv \
#     test_bo_0421/bo_bandwidth_10_12_single_env_const/validation_log.csv \
#     test_bo_0421/bo_bandwidth_10_12_single_env_linear_0.001_12_batch720/validation_log.csv \
#     test_bo_0421/bo_bandwidth_10_12_single_env_linear_0.002_24/validation_log.csv \
#     test_bo_0421/bo_bandwidth_10_12_single_env_linear_0.002_24_batch720/validation_log.csv \
#     test_bo_0421/bo_bandwidth_11_12_single_env/validation_log.csv \
#     test_bo_0421/bo_bandwidth_12_only/validation_log.csv \
#     test_bo_0421/bo_bandwidth_single_env/validation_log.csv \
#     test_bo_0421/bo_bandwidth_single_env_linear/validation_log.csv \
#     test_bo_0421/bo_delay/validation_log.csv \
#     test_bo_0421/bo_T_s_bandwidth/validation_log.csv \
#     test_bo_0421/bo_T_s_bandwidth1/validation_log.csv \
#     test_bo_0421/bo_T_s_bandwidth2/validation_log.csv \
#     test_bo_0421/bo_T_s_bandwidth_min/validation_log.csv \
#     test_bo_0421/bo_T_s_bandwidth_narrow/validation_log.csv \
#     test_bo_0421/bo_T_s_bandwidth_3_4_logscale/validation_log.csv \
#     test_bo_0421/bo_T_s_bandwidth_3_4_logscale_small_bw/validation_log.csv \
#     test_bo_0421/bo_T_s_delay/validation_log.csv \
#     test_bo_0421/bo_T_s_delay_narrow/validation_log.csv \
#     --save-dir test_bo_0421


# python ../plot_scripts/plot_validation.py --log-file \
#     /data/zxxia/PCC-RL/results_0905/udr/real_fail0/validation_log.csv \
#     /data/zxxia/PCC-RL/results_0905/udr/real_fail2/validation_log.csv \
#     /data/zxxia/PCC-RL/results_0905/udr/real_fail5/validation_log.csv \
#     /data/zxxia/PCC-RL/results_0905/udr/real_fail10/validation_log.csv \
#     /data/zxxia/PCC-RL/results_0905/udr/real_fail20/validation_log.csv \
#     --save-dir /data/zxxia/PCC-RL/results_0905/udr

# python ../plot_scripts/plot_validation.py --log-file \
#     /data/zxxia/PCC-RL/results_0905/udr/20mbps/real_fail0/validation_log.csv \
#     /data/zxxia/PCC-RL/results_0905/udr/20mbps/real_fail2/validation_log.csv \
#     /data/zxxia/PCC-RL/results_0905/udr/20mbps/real_fail5/validation_log.csv \
#     /data/zxxia/PCC-RL/results_0905/udr/20mbps/real_fail10/validation_log.csv \
#     /data/zxxia/PCC-RL/results_0905/udr/20mbps/real_fail20/validation_log.csv \
#     --save-dir /data/zxxia/PCC-RL/results_0905/udr
# python ../plot_scripts/plot_validation.py --log-file \
#     /data/zxxia/PCC-RL/results_0905/udr/50mbps/real_fail0/validation_log.csv \
#     /data/zxxia/PCC-RL/results_0905/udr/50mbps/real_fail2/validation_log.csv \
#     /data/zxxia/PCC-RL/results_0905/udr/50mbps/real_fail5/validation_log.csv \
#     /data/zxxia/PCC-RL/results_0905/udr/50mbps/real_fail10/validation_log.csv \
#     /data/zxxia/PCC-RL/results_0905/udr/50mbps/real_fail20/validation_log.csv \
#     --save-dir /data/zxxia/PCC-RL/results_0905/udr

# python ../plot_scripts/plot_validation.py --log-file \
#     /data/zxxia/PCC-RL/results_0905/udr/udr_large_short_queue/seed_20/validation_log.csv \
#     --save-dir /data/zxxia/PCC-RL/results_0905/udr/udr_large_short_queue/seed_20

# for noise in 0 2 5 10 20; do
#     python ../plot_scripts/plot_validation.py --log-file \
#         /data/zxxia/PCC-RL/results_0905/udr/udr_large_short_queue_${noise}/seed_40/validation_log.csv \
#         --save-dir /data/zxxia/PCC-RL/results_0905/udr/udr_large_short_queue_${noise}/seed_40
# done

for bo in 0 1 2 3 4 5 6; do
    python ../plot_scripts/plot_validation.py --log-file \
        /data/zxxia/PCC-RL/results_0910/genet_bbr_noise_udr_start/bo_${bo}/validation_log.csv \
        --save-dir /data/zxxia/PCC-RL/results_0910/genet_bbr_noise_udr_start/bo_${bo} &
    python ../plot_scripts/plot_validation.py --log-file \
        /data/zxxia/PCC-RL/results_0910/genet_bbr_no_noise_udr_start/bo_${bo}/validation_log.csv \
        --save-dir /data/zxxia/PCC-RL/results_0910/genet_bbr_no_noise_udr_start/bo_${bo} &
    python ../plot_scripts/plot_validation.py --log-file \
        /data/zxxia/PCC-RL/results_0910/genet_bbr_no_noise_udr_large_start/bo_${bo}/validation_log.csv \
        --save-dir /data/zxxia/PCC-RL/results_0910/genet_bbr_no_noise_udr_large_start/bo_${bo} &
done
