#!/bin/bash

python ../plot_scripts/plot_validation.py --log-file \
    test_bo_0421/bo_bandwidth/validation_log.csv \
    test_bo_0421/bo_bandwidth_10_12_single_env/validation_log.csv \
    test_bo_0421/bo_bandwidth_10_12_single_env_const/validation_log.csv \
    test_bo_0421/bo_bandwidth_10_12_single_env_linear_0.001_12_batch720/validation_log.csv \
    test_bo_0421/bo_bandwidth_10_12_single_env_linear_0.002_24/validation_log.csv \
    test_bo_0421/bo_bandwidth_10_12_single_env_linear_0.002_24_batch720/validation_log.csv \
    test_bo_0421/bo_bandwidth_11_12_single_env/validation_log.csv \
    test_bo_0421/bo_bandwidth_12_only/validation_log.csv \
    test_bo_0421/bo_bandwidth_single_env/validation_log.csv \
    test_bo_0421/bo_bandwidth_single_env_linear/validation_log.csv \
    test_bo_0421/bo_delay/validation_log.csv \
    test_bo_0421/bo_T_s_bandwidth/validation_log.csv \
    test_bo_0421/bo_T_s_bandwidth1/validation_log.csv \
    test_bo_0421/bo_T_s_bandwidth2/validation_log.csv \
    test_bo_0421/bo_T_s_bandwidth_min/validation_log.csv \
    test_bo_0421/bo_T_s_bandwidth_narrow/validation_log.csv \
    test_bo_0421/bo_T_s_bandwidth_3_4_logscale/validation_log.csv \
    test_bo_0421/bo_T_s_bandwidth_3_4_logscale_small_bw/validation_log.csv \
    test_bo_0421/bo_T_s_delay/validation_log.csv \
    test_bo_0421/bo_T_s_delay_narrow/validation_log.csv \
    --save-dir test_bo_0421

