#!/bin/bash
set -e
for noise in 0; do
    # 0.0001 0.0005 0.001 0.005 0.01 0.015 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.25 0.3 0.35 0.4; do
# for noise in 8 ; do0.5 1 2 3 4 5 6 7 8
    CUDA_VISIBLE_DEVICES="" python evaluate_aurora.py --bandwidth 2 --save-dir test_noise/${noise} --loss 0 --duration 6 --queue 10000 --delay 24 --loss 0 --model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt --noise ${noise}
    python ../plot_scripts/plot_time_series.py --log-file test_noise/${noise}/aurora_simulation_log.csv --save-dir test_noise/${noise} --noise ${noise} >> test_noise/noise_log.csv
done
