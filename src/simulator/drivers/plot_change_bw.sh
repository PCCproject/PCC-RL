
for d_bw in 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1; do
    python evaluate_aurora.py --model-path ../../results_0430/udr_7_dims/bo_delay5/seed_50/model_step_604800.ckpt \
        --save-dir test_aurora1/${d_bw} --trace-file ../../data/synthetic_traces_30s/rand_d_bw/${d_bw}/trace0001.json
    python evaluate_cubic.py --trace-file ../../data/synthetic_traces_30s/rand_d_bw/${d_bw}/trace0001.json \
        --save-dir test_aurora1/${d_bw}
    python ../plot_scripts/plot_packet_log.py --log-file test_aurora1/${d_bw}/aurora_packet_log.csv test_aurora1/${d_bw}/cubic_packet_log.csv --save-dir test_aurora1/${d_bw} \
        --trace-file ../../data/synthetic_traces_30s/rand_d_bw/${d_bw}/trace0001.json
done
