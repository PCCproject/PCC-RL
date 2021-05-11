for bw in 0.6 1; do
    for delay in 5 100; do
        # python evaluate_cubic.py --bandwidth ${bw} --delay ${delay} --duration 30 --save-dir test_aurora/${bw}_${delay}
        # python evaluate_aurora.py --model-path ../../results_0426/udr_7_dims/bo_delay0/model_step_1468800.ckpt --save-dir test_aurora/${bw}_${delay} --bandwidth ${bw} --delay ${delay} --duration 30
        python evaluate_aurora.py --model-path ../../results_0430/udr_7_dims/bo_delay2/seed_50/model_step_1562400.ckpt \
        --save-dir test_aurora/${bw}_${delay} --bandwidth ${bw} --delay ${delay} --duration 30
        python ../plot_scripts/plot_packet_log.py --log-file test_aurora/${bw}_${delay}/aurora_packet_log.csv test_aurora/${bw}_${delay}/cubic_packet_log.csv --save-dir test_aurora/${bw}_${delay} --trace-file trace_${bw}.json
    done
done

