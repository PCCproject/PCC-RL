import argparse
import csv
import glob
import os
from typing import List

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# from plot_scripts.plot_packet_log import PacketLog
from common.utils import compute_std_of_mean
from simulator.trace import Trace

RESULT_ROOT = "test"
TRACE_ROOT = "../../data"
# TRACE_ROOT = "../../data/cellular/2018-12-11T00-27-AWS-Brazil-2-to-Colombia-cellular-3-runs-3-flows"
EXP_NAME2 = "train_perf3"
EXP_NAME1 = "train_perf1"
EXP_NAME = "train_perf2"

TARGET_CCS = ["bbr", "cubic", "vegas", "indigo", "ledbat", "quic"]
# TARGET_CCS = ["bbr"] #, "cubic", "vegas", "indigo", "ledbat", "quic"]


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Aurora Testing in simulator.")
    parser.add_argument('--save-dir', type=str, default="",
                        help="direcotry to testing results.")
    # parser.add_argument('--cc', type=str, required=True,
    #                     choices=("bbr", "cubic", "udr1", "udr2", "udr3",
    #                              "genet_bbr", 'genet_cubic'),
    #                     help='congestion control name')
    parser.add_argument("--conn-type", type=str,
                        choices=('ethernet', 'cellular', 'wifi'),
                        help='connection type')

    args, _ = parser.parse_known_args()
    return args


def load_cc_rewards_across_traces(log_files: List[str]) -> List[float]:
    rewards = []
    for log_file in log_files:
        if not os.path.exists(log_file):
            print(log_file, 'does not exist')
            continue
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rewards.append(float(row['pkt_level_reward']))
        # pkt_log = PacketLog.from_log_file(log_file)
        # rewards.append(pkt_log.get_reward("", trace))
    return rewards


def main():
    args = parse_args()
    link_dirs = glob.glob(os.path.join(TRACE_ROOT, args.conn_type, "*/"))
    if args.conn_type == 'ethernet':
        queue_size = 500
    elif args.conn_type == 'cellular':
        queue_size = 50
    else:
        raise ValueError
    traces = []
    save_dirs = []
    for link_dir in link_dirs:
        link_name = link_dir.split('/')[-2]
        print(args.conn_type, link_name)
        if link_name != "2019-09-17T22-29-AWS-California-1-to-Stanford-cellular-3-runs":
            continue
        for cc in TARGET_CCS:
            print("Loading real traces collected by {}...".format(cc))
            for trace_file in tqdm(sorted(glob.glob(os.path.join(link_dir, "{}_datalink_run[1,3].log".format(cc))))):
                traces.append(Trace.load_from_pantheon_file(
                    trace_file, 0.0, queue_size, front_offset=5))
                save_dir = os.path.join(args.save_dir, args.conn_type, link_name,
                                        os.path.splitext(os.path.basename(trace_file))[0])
                save_dirs.append(save_dir)

    bbr_old_rewards = load_cc_rewards_across_traces(
        [os.path.join(save_dir, "bbr_old", "bbr_old_summary.csv") for save_dir in save_dirs])
    bbr_rewards = load_cc_rewards_across_traces(
        [os.path.join(save_dir, "bbr", "bbr_summary.csv") for save_dir in save_dirs])
    cubic_rewards = load_cc_rewards_across_traces(
        [os.path.join(save_dir, "cubic", "cubic_summary.csv") for save_dir in save_dirs])
    print(bbr_rewards)
    steps = []
    genet_bbr_rewards = []
    genet_bbr_old_rewards = []
    genet_cubic_rewards = []
    for bo in range(0, 30, 3):
        genet_seed = 42
        tmp_rewards = load_cc_rewards_across_traces(
            [os.path.join(save_dir, 'genet_bbr', 'bo_{}'.format(bo), "seed_{}".format(genet_seed), 'step_64800', 'aurora_summary.csv') for save_dir in save_dirs])
        # [os.path.join(save_dir, 'genet_bbr', 'bo_{}'.format(bo), 'step_64800', 'aurora_summary.csv') for save_dir in save_dirs])
        steps.append(bo * 72000)
        print(tmp_rewards)
        genet_bbr_rewards.append(np.mean(tmp_rewards))
        tmp_rewards = load_cc_rewards_across_traces(
            [os.path.join(save_dir, 'genet_cubic', 'bo_{}'.format(bo), "seed_{}".format(genet_seed), 'step_64800', 'aurora_summary.csv') for save_dir in save_dirs])
        # [os.path.join(save_dir, 'genet_cubic', 'bo_{}'.format(bo), 'step_64800','aurora_summary.csv') for save_dir in save_dirs])
        print(save_dirs)
        print(tmp_rewards)
        genet_cubic_rewards.append(np.mean(tmp_rewards))

        tmp_rewards = load_cc_rewards_across_traces(
            [os.path.join(save_dir, 'genet_bbr_old', "seed_{}".format(genet_seed),
                          'bo_{}'.format(bo),  'step_64800', 'aurora_summary.csv') for save_dir in save_dirs])
        # [os.path.join(save_dir, 'genet_bbr', 'bo_{}'.format(bo), 'step_64800', 'aurora_summary.csv') for save_dir in save_dirs])
        print(tmp_rewards)
        genet_bbr_old_rewards.append(np.mean(tmp_rewards))

    udr1_avg_rewards = []
    udr2_avg_rewards = []
    udr3_avg_rewards = []
    udr1_reward_errs = []
    udr2_reward_errs = []
    udr3_reward_errs = []
    udr_steps = [0, 43200, 100800, 158400, 216000,
                 259200, 316800, 374400, 432000, 48900]
    for step in udr_steps:
        step = int(step)

        udr1_avg_rewards_across_models = []
        udr2_avg_rewards_across_models = []
        udr3_avg_rewards_across_models = []
        for seed in tqdm(range(10, 110, 10)):
            udr1_rewards = load_cc_rewards_across_traces([os.path.join(
                save_dir, "udr1", "seed_{}".format(
                    seed), "step_{}".format(step),
                'aurora_packet_log.csv') for save_dir in save_dirs])
            if udr1_rewards:
                udr1_avg_rewards_across_models.append(np.mean(udr1_rewards))

            udr2_rewards = load_cc_rewards_across_traces([os.path.join(
                save_dir, "udr2", "seed_{}".format(
                    seed), "step_{}".format(step),
                'aurora_packet_log.csv') for save_dir in save_dirs])
            if udr2_rewards:
                udr2_avg_rewards_across_models.append(np.mean(udr2_rewards))

            udr3_rewards = load_cc_rewards_across_traces([os.path.join(
                save_dir, "udr3",  "seed_{}".format(
                    seed), "step_{}".format(step),
                'aurora_packet_log.csv') for save_dir in save_dirs])
            if udr3_rewards:
                udr3_avg_rewards_across_models.append(np.mean(udr3_rewards))
        udr1_avg_rewards.append(np.mean(udr1_avg_rewards_across_models))
        udr2_avg_rewards.append(np.mean(udr2_avg_rewards_across_models))
        udr3_avg_rewards.append(np.mean(udr3_avg_rewards_across_models))
        udr1_reward_errs.append(compute_std_of_mean(
            udr1_avg_rewards_across_models))
        udr2_reward_errs.append(compute_std_of_mean(
            udr2_avg_rewards_across_models))
        udr3_reward_errs.append(compute_std_of_mean(
            udr3_avg_rewards_across_models))

    udr1_low_bnd = np.array(udr1_avg_rewards) - np.array(udr1_reward_errs)
    udr1_up_bnd = np.array(udr1_avg_rewards) + np.array(udr1_reward_errs)
    udr2_low_bnd = np.array(udr2_avg_rewards) - np.array(udr2_reward_errs)
    udr2_up_bnd = np.array(udr2_avg_rewards) + np.array(udr2_reward_errs)
    udr3_low_bnd = np.array(udr3_avg_rewards) - np.array(udr3_reward_errs)
    udr3_up_bnd = np.array(udr3_avg_rewards) + np.array(udr3_reward_errs)

    plt.axhline(y=np.mean(bbr_rewards), ls="--", label="BBR")
    plt.axhline(y=np.mean(bbr_old_rewards), ls="-.", label="BBR old")
    plt.axhline(y=np.mean(cubic_rewards), ls=":", label="Cubic")
    plt.plot(steps, genet_bbr_rewards, "-.", label='GENET_BBR')
    plt.plot(steps, genet_cubic_rewards, "-", label='GENET_Cubic')
    plt.plot(steps, genet_bbr_old_rewards, "-", label='GENET_BBR_old')
    print(genet_bbr_old_rewards)

    # plt.plot(udr_steps, udr1_avg_rewards, "-", label='UDR-1')
    # plt.fill_between(steps, udr1_low_bnd, udr1_up_bnd, color='grey', alpha=0.1)
    # plt.plot(udr_steps, udr2_avg_rewards, "--", label='UDR-2')
    # plt.fill_between(steps, udr2_low_bnd, udr2_up_bnd, color='grey', alpha=0.1)
    # print(steps, genet_bbr_rewards)
    # print(steps, genet_cubic_rewards)
    # print(udr_steps, udr3_avg_rewards)
    # assert len(udr_steps) == len(udr3_avg_rewards)
    # plt.plot(udr_steps, udr3_avg_rewards, "-.", label='UDR-3')
    # plt.fill_between(udr_steps, udr3_low_bnd, udr3_up_bnd, color='grey', alpha=0.1)
    plt.legend()
    plt.show()
    # plt.savefig('train_curve.jpg')


if __name__ == '__main__':
    main()
