import os
import numpy as np
from utils import read_json_file

ROOT = '../../results/tmp'

bw_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for i, bw in zip(range(10), bw_list):
    log = read_json_file(os.path.join(ROOT, 'pcc_env_log_run_{}.json'.format(i)))
    reward = [event['Reward'] for event in log['Events']]
    throughput = [event['Throughput'] for event in log['Events']]
    latency = [event['Latency'] for event in log['Events']]
    loss_rate = [event['Loss Rate']for event in log['Events']]
    latency_inflation = [event['Latency Inflation'] for event in log['Events']]
    print(i, np.mean(reward), np.mean(throughput), np.mean(latency),
          np.mean(loss_rate), np.mean(latency_inflation), bw, 0.05, 0, 5)
