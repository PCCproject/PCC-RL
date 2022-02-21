import os

import pandas as pd

from common.utils import read_json_file, set_seed

from simulator.aurora import test_on_traces
from simulator.trace import generate_traces
from simulator.network_simulator.bbr_old import BBR_old

set_seed(42)
ROOT = "/tank/zxxia/PCC-RL/results_0928/genet_no_reward_scale/genet_bbr_old"
ROOT = "/datamirror/zxxia/PCC-RL/results_1006/gap_vs_improvement/train"
# SAVE_ROOT = "/datamirror/zxxia/PCC-RL/results_1006/gap_vs_improvement/test_50traces"
SAVE_ROOT = "/datamirror/zxxia/PCC-RL/results_1006/gap_vs_improvement/test"
MODEL_PATH_BEFORE = "/tank/zxxia/PCC-RL/results_0928/genet_no_reward_scale/genet_bbr_old/seed_10/bo_0/model_step_64800.ckpt"
TRACE_CNT=10
# TRACE_CNT=50

# ROOT = "/datamirror/zxxia/PCC-RL/results_1006/gap_vs_improvement_pretrained/train"
# SAVE_ROOT = "/datamirror/zxxia/PCC-RL/results_1006/gap_vs_improvement_pretrained/test"
# MODEL_PATH_BEFORE = "../../results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt"
seed = 10
#
for config_id in range(100, 110):
#     if bo == 0:
#         model_path_before = "/tank/zxxia/PCC-RL/results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt"
#     else:
#         model_path_before = os.path.join(ROOT, 'seed_{}'.format(seed), 'bo_{}'.format(bo-1), 'model_step_64800.ckpt')
    # model_path_after = os.path.join(ROOT, 'seed_{}'.format(seed), 'config_{:02d}'.format(config_id), 'model_step_64800.ckpt')
    val_log_path = os.path.join(ROOT, 'seed_{}'.format(seed), 'config_{:02d}'.format(config_id), 'validation_log.csv')
    if not os.path.exists(val_log_path):
        print('skip', val_log_path)
        continue
    val_log = pd.read_csv(val_log_path, delimiter='\t')
    # idx = val_log['mean_validation_reward'].argmax()
    idx = val_log['mean_validation_pkt_level_reward'].argmax()
    model_path_after = os.path.join(ROOT, 'seed_{}'.format(seed), 'config_{:02d}'.format(config_id), 'model_step_{}.ckpt'.format(int(val_log['num_timesteps'][idx])))
    if not os.path.exists(model_path_after+'.index') or not os.path.exists(os.path.join(ROOT, 'seed_{}'.format(seed), 'config_{:02d}'.format(config_id), 'model_step_64800.ckpt.meta')):
        print('skip', model_path_after)
        continue
#     config = read_json_file(os.path.join(ROOT, 'seed_{}'.format(seed), "bo_{}.json".format(bo)))[-1]
#     config['weight'] = 1
#     config = [config]
    # config_file = '../../config/gap_vs_improvement_pretrained/config_{:02d}.json'.format(config_id)
    config_file = '../../config/gap_vs_improvement/config_{:02d}.json'.format(config_id)
    traces = generate_traces(config_file, TRACE_CNT, 30)
    save_dirs = [os.path.join(SAVE_ROOT, 'seed_{}'.format(seed),
        'config_{:02d}'.format(config_id), 'trace_{:05d}'.format(i)) for i in range(len(traces))]
    before_save_dirs = [os.path.join(save_dir, 'before') for save_dir in save_dirs]
    after_save_dirs = [os.path.join(save_dir, 'after_best_pkt_level_reward') for save_dir in save_dirs]
    bbr_old_save_dirs = [os.path.join(save_dir, 'bbr_old') for save_dir in save_dirs]
    if os.path.exists(os.path.join(after_save_dirs[0], 'aurora_summary.csv')):
        continue

    test_on_traces(MODEL_PATH_BEFORE, traces, before_save_dirs, 16, seed, False, True)
    test_on_traces(model_path_after, traces, after_save_dirs, 16, seed, False, True)
    after_save_dirs = [os.path.join(save_dir, 'after') for save_dir in save_dirs]
    model_path_after = os.path.join(ROOT, 'seed_{}'.format(seed), 'config_{:02d}'.format(config_id), 'model_step_{}.ckpt'.format(64800))
    test_on_traces(model_path_after, traces, after_save_dirs, 16, seed, False, True)

    baseline = BBR_old(False)
    baseline.test_on_traces(traces, bbr_old_save_dirs, plot_flag=True, n_proc=16)

    for i, (trace, save_dir) in enumerate(zip(traces, save_dirs)):
        trace.dump(os.path.join(save_dir, "trace_{:05d}.json".format(i)))
