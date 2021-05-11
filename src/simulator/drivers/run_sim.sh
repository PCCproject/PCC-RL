
# python evaluate_aurora.py --trace-file ../../data/synthetic_traces/rand_delay_noise/0.0/trace0000.json --model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt --save-dir test_aurora4
# python ../plot_scripts/plot_time_series.py --log-file test_aurora4/aurora_simulation_log.csv --save-dir test_aurora4 --trace-file ../../data/synthetic_traces/rand_delay_noise/0.0/trace0000.json

# python evaluate_aurora.py --trace-file ../../data/synthetic_traces/rand_delay_noise/0.0/trace0000.json --model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt --save-dir test_aurora3
# python ../plot_scripts/plot_time_series.py --log-file test_aurora3/aurora_simulation_log.csv --save-dir test_aurora3 --trace-file ../../data/synthetic_traces/rand_delay_noise/0.0/trace0000.json

# python evaluate_aurora.py --trace-file ../../data/synthetic_traces/rand_delay_noise/0.0/trace0000.json --model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt --save-dir test_aurora2
# python ../plot_scripts/plot_time_series.py --log-file test_aurora2/aurora_simulation_log.csv --save-dir test_aurora2 --trace-file ../../data/synthetic_traces/rand_delay_noise/0.0/trace0000.json



python evaluate_aurora.py --trace-file ../../data/synthetic_traces/rand_delay_noise/0.0/trace0000.json --model-path ../../results_0415/udr_7_dims_fix_val_reward/range0_queue50/model_step_252000.ckpt --save-dir test_aurora0
python ../plot_scripts/plot_time_series.py --log-file test_aurora0/aurora_simulation_log.csv --save-dir test_aurora0 --trace-file ../../data/synthetic_traces/rand_delay_noise/0.0/trace0000.json
