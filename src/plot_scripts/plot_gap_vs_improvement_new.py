import os

import matplotlib.pyplot as plt
import numpy as np

from common.utils import load_summary, load_bo_json_log, read_json_file


ROOT = "/tank/zxxia/PCC-RL/results_0928/genet_no_reward_scale/genet_bbr_old"
SAVE_ROOT = "/datamirror/zxxia/PCC-RL/results_1006/gap_vs_improvement/test"
CONFIG_ROOT = "../../config/gap_vs_improvement"



def load_results(save_dirs, cc='aurora'):
    rewards = []
    for save_dir in save_dirs:
        if not os.path.exists(os.path.join(save_dir, '{}_summary.csv'.format(cc))):
            continue
        summary = load_summary(os.path.join(save_dir, '{}_summary.csv'.format(cc)))
        rewards.append(summary['pkt_level_reward'])
    return np.mean(rewards), np.std(rewards)

def get_max_gap(bo_logs):
    gaps = [bo_log['target'] for bo_log in bo_logs]
    idx = np.argmax(gaps)
    return max(gaps), bo_logs[idx]['params']

gaps = []
improvements = []
seed = 10
for config_id in range(90):
    config = read_json_file(os.path.join(CONFIG_ROOT, 'config_{:02d}.json'.format(config_id)))[0]
    save_dirs = [os.path.join(SAVE_ROOT, 'seed_{}'.format(seed),
        'config_{:02d}'.format(config_id), 'trace_{:05d}'.format(i)) for i in range(10)]
    before_save_dirs = [os.path.join(save_dir, 'before') for save_dir in save_dirs]
    after_save_dirs = [os.path.join(save_dir, 'after_best_pkt_level_reward') for save_dir in save_dirs]
    bbr_save_dirs = [os.path.join(save_dir, 'bbr_old') for save_dir in save_dirs]
    reward_before, reward_std_before = load_results(before_save_dirs)
    reward_after, reward_std_after = load_results(after_save_dirs)
    bbr_old_reward, reward_std_bbr_old = load_results(bbr_save_dirs, 'bbr_old')

    improvement = reward_after - reward_before
    gap = bbr_old_reward - reward_before

    if gap < 0 or config_id == 14:
        continue
    gaps.append(gap)
    improvements.append(improvement)
    # if gap > 0:
    #     print("config_id: {}, gap={:.2f}, improv={:.2f}".format(config_id, gap, improvement))
    if gap > 50 and gap < 100:
        print("config_id: {}, gap={:.2f}, improv={:.2f}, max_bw={:.2f}, delay={:.2f}, T_s={:.2f}, queue={:.2f}".format(
            config_id, gap, improvement, config['bandwidth_upper_bound'][0], config['delay'][0], config['T_s'][0], config['queue'][0]))
    elif gap > 150 and gap < 250:
        print("strange config_id: {}, gap={:.2f}, improv={:.2f}, max_bw={:.2f}, delay={:.2f}, T_s={:.2f}, queue={:.2f}".format(
            config_id, gap, improvement, config['bandwidth_upper_bound'][0], config['delay'][0], config['T_s'][0], config['queue'][0]))
    elif gap > 100 and gap < 110:
        print("strange1 config_id: {}, gap={:.2f}, improv={:.2f}, max_bw={:.2f}, delay={:.2f}, T_s={:.2f}, queue={:.2f}".format(
            config_id, gap, improvement, config['bandwidth_upper_bound'][0], config['delay'][0], config['T_s'][0], config['queue'][0]))

# print(improvements)
#     if improvement < 0:
#         print(gap, improvement, params, bo, seed)
# m, b = np.polyfit(gaps, improvements, 1)
# plt.plot(np.arange(300), m*np.arange(300)+b)
print(len(gaps))
plt.scatter(gaps, improvements)
plt.axhline(y=0, c='k', ls='--')
plt.xlabel('Gap(BBR - Aurora reward)')
plt.ylabel('Improvement(after training - before training)')
plt.show()
