#!/bin/bash

set -e

# python trace.py --config-file ../../config/test/fig8_7_dims/rand_T_s_bandwidth.json --save-dir rand_T_s_bandwidth
# python trace.py --config-file ../../config/test/fig8_7_dims/rand_bandwidth.json --save-dir rand_bandwidth
# python trace.py --config-file ../../config/test/fig8_7_dims/rand_T_s_delay.json --save-dir rand_T_s_delay
# python trace.py --config-file ../../config/test/fig8_7_dims/rand_delay.json --save-dir rand_delay
# python trace.py --config-file ../../config/test/fig8_7_dims/rand_loss.json --save-dir rand_loss
# python trace.py --config-file ../../config/test/fig8_7_dims/rand_queue.json --save-dir rand_queue
# python trace.py --config-file ../../config/test/fig8_7_dims/rand_ack_delay_prob.json --save-dir rand_ack_delay_prob

# save_dir=bw_vary_traces
# save_dir=bw_vary_traces_2mbps_new
save_dir=tests
# config_dir=../../config/test/fig8_7_dims_2mbps
config_dir=../../config/train/udr_7_dims_0422

python trace.py --config-file ${config_dir}/range0.json --save-dir ${save_dir}/rand_T_s_bandwidth
# python trace.py --config-file ${config_dir}/rand_T_s_bandwidth.json --save-dir ${save_dir}/rand_T_s_bandwidth
# python trace.py --config-file ${config_dir}/rand_bandwidth.json --save-dir ${save_dir}/rand_bandwidth
# python trace.py --config-file ${config_dir}/rand_duration.json --save-dir ${save_dir}/rand_duration
# python trace.py --config-file ${config_dir}/rand_T_s_delay.json --save-dir ${save_dir}/rand_T_s_delay
# python trace.py --config-file ${config_dir}/rand_delay.json --save-dir ${save_dir}/rand_delay
# python trace.py --config-file ${config_dir}/rand_loss.json --save-dir ${save_dir}/rand_loss
# python trace.py --config-file ${config_dir}/rand_queue.json --save-dir ${save_dir}/rand_queue
# python trace.py --config-file ${config_dir}/rand_ack_delay_prob.json --save-dir ${save_dir}/rand_ack_delay_prob
