root=check1
python evaluate_aurora.py --model-path ../../results_0426/udr_7_dims/bo_delay0/model_step_1468800.ckpt \
    --save-dir ${root}/low_bw_low_rtt --trace-file ${root}/low_bw_low_rtt.json

python evaluate_aurora.py --model-path ../../results_0426/udr_7_dims/bo_delay0/model_step_1468800.ckpt \
    --save-dir ${root}/low_bw_high_rtt --trace-file ${root}/low_bw_high_rtt.json

python evaluate_cubic.py --trace-file ${root}/low_bw_low_rtt.json --save-dir ${root}/low_bw_low_rtt
python evaluate_cubic.py --trace-file ${root}/low_bw_high_rtt.json --save-dir ${root}/low_bw_high_rtt

python ../plot_scripts/plot_time_series.py \
    --log-file ${root}/low_bw_low_rtt/aurora_simulation_log.csv ${root}/low_bw_low_rtt/cubic_simulation_log.csv \
    --save-dir ${root}/low_bw_low_rtt --trace-file ${root}/low_bw_low_rtt

python ../plot_scripts/plot_time_series.py \
    --log-file ${root}/low_bw_high_rtt/aurora_simulation_log.csv ${root}/low_bw_high_rtt/cubic_simulation_log.csv \
    --save-dir ${root}/low_bw_high_rtt --trace-file ${root}/low_bw_high_rtt
