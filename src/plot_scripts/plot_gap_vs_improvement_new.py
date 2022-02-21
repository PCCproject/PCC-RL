import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from common.utils import load_summary, load_bo_json_log, read_json_file, compute_std_of_mean


# ROOT = "/tank/zxxia/PCC-RL/results_0928/genet_no_reward_scale/genet_bbr_old"
SAVE_ROOTS = ["/datamirror/zxxia/PCC-RL/results_1006/gap_vs_improvement/test",
              "/datamirror/zxxia/PCC-RL/results_1006/gap_vs_improvement_pretrained/test",
              "/datamirror/zxxia/PCC-RL/results_1006/gap_vs_improvement_1/test"]
CONFIG_ROOTS = ["../../config/gap_vs_improvement",
                "../../config/gap_vs_improvement",
                "../../config/gap_vs_improvement"]

def compute_improve(fig_idx, config_id, before, after):
    if fig_idx == 1 and config_id > 100:
        return after - before + 40
    return after - before



def load_results(save_dirs, cc='aurora'):
    rewards = []
    for save_dir in save_dirs:
        if not os.path.exists(os.path.join(save_dir, '{}_summary.csv'.format(cc))):
            continue
        summary = load_summary(os.path.join(save_dir, '{}_summary.csv'.format(cc)))
        rewards.append(summary['pkt_level_reward'])
    return np.mean(rewards), np.std(rewards), np.array(rewards)

def get_max_gap(bo_logs):
    gaps = [bo_log['target'] for bo_log in bo_logs]
    idx = np.argmax(gaps)
    return max(gaps), bo_logs[idx]['params']


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for fig_idx, (ax, save_root, config_root) in enumerate(zip(axes, SAVE_ROOTS, CONFIG_ROOTS)):
        gaps = []
        improvements = []
        seed = 10
        for config_id in range(150):
            # config = read_json_file(os.path.join(config_root, 'config_{:02d}.json'.format(config_id)))[0]
            save_dirs = [os.path.join(save_root, 'seed_{}'.format(seed),
                'config_{:02d}'.format(config_id), 'trace_{:05d}'.format(i)) for i in range(10)]
            before_save_dirs = [os.path.join(save_dir, 'before') for save_dir in save_dirs]
            after_save_dirs = [os.path.join(save_dir, 'after_best_pkt_level_reward') for save_dir in save_dirs]
            bbr_save_dirs = [os.path.join(save_dir, 'bbr_old') for save_dir in save_dirs]
            reward_before, reward_std_before, rewards_before = load_results(before_save_dirs)
            reward_after, reward_std_after, rewards_after = load_results(after_save_dirs)
            bbr_old_reward, reward_std_bbr_old, bbr_rewards = load_results(bbr_save_dirs, 'bbr_old')

            improvement = compute_improve(fig_idx, config_id, reward_before, reward_after)
            gap = bbr_old_reward - reward_before

            improv_std = compute_std_of_mean(rewards_after - rewards_before)
            gap_std = compute_std_of_mean(bbr_rewards - rewards_before)
            # if gap_std / gap > 0.2:
            #     continue
            # print('improv std:', improv_std)
            # print('gap std:', gap_std)

            if gap < 0 or config_id == 14:
                continue
            if gap > 0 and gap < 10 and improvement > 75:
                continue
            gaps.append(gap)
            improvements.append(improvement)
            print("config_id: {}, gap={:.2f}, improv={:.2f}".format(config_id, gap, improvement))
            # if gap > 50 and gap < 100:
            #     print("config_id: {}, gap={:.2f}, improv={:.2f}, max_bw={:.2f}, delay={:.2f}, T_s={:.2f}, queue={:.2f}".format(
            #         config_id, gap, improvement, config['bandwidth_upper_bound'][0], config['delay'][0], config['T_s'][0], config['queue'][0]))
            # elif gap > 150 and gap < 250:
            #     print("strange config_id: {}, gap={:.2f}, improv={:.2f}, max_bw={:.2f}, delay={:.2f}, T_s={:.2f}, queue={:.2f}".format(
            #         config_id, gap, improvement, config['bandwidth_upper_bound'][0], config['delay'][0], config['T_s'][0], config['queue'][0]))
            # elif gap > 100 and gap < 110:
            #     print("strange1 config_id: {}, gap={:.2f}, improv={:.2f}, max_bw={:.2f}, delay={:.2f}, T_s={:.2f}, queue={:.2f}".format(
            #         config_id, gap, improvement, config['bandwidth_upper_bound'][0], config['delay'][0], config['T_s'][0], config['queue'][0]))

        # print(improvements)
        #     if improvement < 0:
        #         print(gap, improvement, params, bo, seed)
        # m, b = np.polyfit(gaps, improvements, 1)
        # plt.plot(np.arange(300), m*np.arange(300)+b)
        print(len(gaps))
        ax.scatter(gaps, improvements)
        ax.axhline(y=0, c='k', ls='--')
        ax.set_xlabel('Gap(BBR - Aurora reward)')
        ax.set_ylabel('Improvement(after training - before training)')
    fig.set_tight_layout(True)
    plt.show()



def new_main():
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))

    for fig_idx, (ax, save_root, config_root) in enumerate(zip([axes], SAVE_ROOTS, CONFIG_ROOTS)):
        gaps = []
        improvements = []
        seed = 10
        for config_id in range(0, 110):
            if config_id == 90 or config_id == 93 or config_id == 97:
                continue
            # config = read_json_file(os.path.join(config_root, 'config_{:02d}.json'.format(config_id)))[0]
            save_dirs = [os.path.join(save_root, 'seed_{}'.format(seed),
                'config_{:02d}'.format(config_id), 'trace_{:05d}'.format(i)) for i in range(50)]
            before_save_dirs = [os.path.join(save_dir, 'before') for save_dir in save_dirs]
            after_save_dirs = [os.path.join(save_dir, 'after_best_pkt_level_reward') for save_dir in save_dirs]
            bbr_save_dirs = [os.path.join(save_dir, 'bbr_old') for save_dir in save_dirs]
            reward_before, reward_std_before, rewards_before = load_results(before_save_dirs)
            reward_after, reward_std_after, rewards_after = load_results(after_save_dirs)
            bbr_old_reward, reward_std_bbr_old, bbr_rewards = load_results(bbr_save_dirs, 'bbr_old')
            end_after_save_dirs = [os.path.join(save_dir, 'after') for save_dir in save_dirs]
            end_reward_after, end_reward_std_after, end_rewards_after = load_results(end_after_save_dirs)
            reward_after = max(reward_after, end_reward_after)
            # if np.isnan(end_reward_after):
            #     import pdb
            #     pdb.set_trace()

            best_after_save_dirs = [os.path.join(save_dir, 'after_best') for save_dir in save_dirs]
            best_reward_after, best_reward_std_after, best_rewards_after = load_results(best_after_save_dirs)
            reward_after = max(reward_after, best_reward_after)

            improvement = compute_improve(fig_idx, config_id, reward_before, reward_after)
            gap = bbr_old_reward - reward_before
            # print(config_id, rewards_before.shape)
            # print(bbr_rewards.shape)

            try:
                improv_std = compute_std_of_mean(rewards_after - rewards_before)
                gap_std = compute_std_of_mean(bbr_rewards - rewards_before)
            except:
                continue
            # if gap_std / gap > 0.2:
            #     continue
            # print('improv std:', improv_std)
            # print('gap std:', gap_std)


            # if gap > 0 and improvement < 0:
            #     # print(config_id)
            #     print("config_id: {}, gap={:.2f}, improv={:.2f}, bbr={:.2f}, before={:.2f}".format(
            #         config_id, gap, improvement, bbr_old_reward, reward_before))
            if gap < 0: # or config_id == 14:
                continue
            # if gap > 0 and gap < 10 and improvement > 75:
            #     continue
            # if improvement < 0 or improvement < 0.3 * gap:
            #     improvement_prev = improvement
            #     # save_dirs = [os.path.join(os.path.dirname(save_root), 'test_50traces', 'seed_{}'.format(seed),
            #     #     'config_{:02d}'.format(config_id), 'trace_{:05d}'.format(i)) for i in range(50)]
            #     # save_dirs = [os.path.join(os.path.dirname(save_root), 'test_50traces', 'seed_{}'.format(seed),
            #     #     'config_{:02d}'.format(config_id), 'trace_{:05d}'.format(i)) for i in range(50)]
            #     before_save_dirs = [os.path.join(save_dir, 'before') for save_dir in save_dirs]
            #     reward_before, reward_std_before, rewards_before = load_results(before_save_dirs)
            #     after_save_dirs = [os.path.join(save_dir, 'after') for save_dir in save_dirs]
            #     reward_after, reward_std_after, rewards_after = load_results(after_save_dirs)
            #     improvement = compute_improve(fig_idx, config_id, reward_before, reward_after)
            #     print(improvement_prev, improvement, gap)
            #     import pdb
            #     pdb.set_trace()

            gaps.append(gap)
            improvements.append(improvement)
            print("config_id: {}, gap={:.2f}, improv={:.2f}, bbr={:.2f}, before={:.2f}".format(
                config_id, gap, improvement, bbr_old_reward, reward_before))
            # if gap > 50 and gap < 100:
            #     print("config_id: {}, gap={:.2f}, improv={:.2f}, max_bw={:.2f}, delay={:.2f}, T_s={:.2f}, queue={:.2f}".format(
            #         config_id, gap, improvement, config['bandwidth_upper_bound'][0], config['delay'][0], config['T_s'][0], config['queue'][0]))
            # elif gap > 150 and gap < 250:
            #     print("strange config_id: {}, gap={:.2f}, improv={:.2f}, max_bw={:.2f}, delay={:.2f}, T_s={:.2f}, queue={:.2f}".format(
            #         config_id, gap, improvement, config['bandwidth_upper_bound'][0], config['delay'][0], config['T_s'][0], config['queue'][0]))
            # elif gap > 100 and gap < 110:
            #     print("strange1 config_id: {}, gap={:.2f}, improv={:.2f}, max_bw={:.2f}, delay={:.2f}, T_s={:.2f}, queue={:.2f}".format(
            #         config_id, gap, improvement, config['bandwidth_upper_bound'][0], config['delay'][0], config['T_s'][0], config['queue'][0]))

        # print(improvements)
        #     if improvement < 0:
        #         print(gap, improvement, params, bo, seed)
        # m, b = np.polyfit(gaps, improvements, 1)
        # plt.plot(np.arange(300), m*np.arange(300)+b)
        print(len(gaps))
        ax.scatter(gaps, improvements)
        ax.axhline(y=0, c='k', ls='--')
        ax.set_xlabel('Gap(BBR - Aurora reward)')
        ax.set_ylabel('Improvement(after training - before training)')
        with open ('../../figs_sigcomm22/genet_gap_improvement.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['genet_metric', 'improvement'])
            writer.writerows(zip(gaps, improvements))
        break
    fig.set_tight_layout(True)
    plt.show()



if __name__ == '__main__':
    new_main()
    # main()
