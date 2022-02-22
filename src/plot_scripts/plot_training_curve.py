import argparse
import os
from typing import List, Tuple

from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common.utils import compute_std_of_mean, load_summary
from simulator.pantheon_dataset import PantheonDataset

TRACE_ROOT = "../../data"

TARGET_CCS = ["bbr", "cubic", "vegas", "indigo", "ledbat", "quic"]
PLOT_CUBIC = False


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Plot the performance curves on real "
                                     "Pantheon traces over training.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to testing results.")
    parser.add_argument("--conn-type", type=str, default='all',
                        choices=('cellular', 'ethernet', 'wifi', 'all'),
                        help='connection type')

    args, _ = parser.parse_known_args()
    return args


def load_summaries_across_traces(log_files: List[str]) -> Tuple[List[float], List[float], List[float], List[float]]:
    rewards = []
    tputs = []
    lats = []
    losses = []
    for log_file in log_files:
        if not os.path.exists(log_file):
            # print(log_file, 'does not exist')
            continue
        summary = load_summary(log_file)
        rewards.append(summary['pkt_level_reward'])
        tputs.append(summary['average_throughput'])
        lats.append(summary['average_latency'] * 1000)
        losses.append(summary['loss_rate'])
    return rewards, tputs, lats, losses


def contains_nan_only(l) -> bool:
    for v in l:
        if not np.isnan(v):
            return False
    return True

def load_results(save_dirs, seeds, steps, name: str):
    rewards, tputs, lats, losses = [], [], [], []
    reward_errs, tput_errs, lat_errs, loss_errs = [], [], [], []
    for step in steps:
        step = int(step)
        avg_rewards_across_seeds = []
        avg_tputs_across_seeds = []
        avg_lats_across_seeds = []
        avg_losses_across_seeds = []

        for seed in seeds:

            if name == 'real':
                tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = load_summaries_across_traces([os.path.join(
                    save_dir, name, "seed_{}".format(seed),
                    'aurora_summary.csv') for save_dir in save_dirs])
            else:
                tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = load_summaries_across_traces([os.path.join(
                    save_dir, name, "seed_{}".format(seed), "step_{}".format(step),
                    'aurora_summary.csv') for save_dir in save_dirs])
            avg_rewards_across_seeds.append(np.nanmean(np.array(tmp_rewards)))
            avg_tputs_across_seeds.append(np.nanmean(np.array(tmp_tputs)))
            avg_lats_across_seeds.append(np.nanmean(np.array(tmp_lats)))
            avg_losses_across_seeds.append(np.nanmean(np.array(tmp_losses)))
            # if name == 'real':
            #     import pdb
            #     pdb.set_trace()
            if name == 'cl2_new':
                print(len(tmp_rewards), np.mean(tmp_rewards))

        rewards.append(np.nanmean(np.array(avg_rewards_across_seeds)))
        reward_errs.append(compute_std_of_mean(avg_rewards_across_seeds))

        tputs.append(np.nanmean(np.array(avg_tputs_across_seeds)))
        tput_errs.append(compute_std_of_mean(avg_tputs_across_seeds))

        lats.append(np.nanmean(np.array(avg_lats_across_seeds)))
        lat_errs.append(compute_std_of_mean(avg_lats_across_seeds))

        losses.append(np.nanmean(np.array(avg_losses_across_seeds)))
        loss_errs.append(compute_std_of_mean(avg_losses_across_seeds))

    low_bnd = np.array(rewards) - np.array(reward_errs)
    up_bnd = np.array(rewards) + np.array(reward_errs)

    tputs_low_bnd = np.array(tputs) - np.array(tput_errs)
    tputs_up_bnd = np.array(tputs) + np.array(tput_errs)

    lats_low_bnd = np.array(lats) - np.array(lat_errs)
    lats_up_bnd = np.array(lats) + np.array(lat_errs)

    losses_low_bnd = np.array(losses) - np.array(loss_errs)
    losses_up_bnd = np.array(losses) + np.array(loss_errs)
    return (rewards, tputs, lats, losses, low_bnd, up_bnd,
            tputs_low_bnd, tputs_up_bnd, lats_low_bnd, lats_up_bnd,
            losses_low_bnd, losses_up_bnd)


def main():
    args = parse_args()
    if args.conn_type == 'all':
        cellular_dataset = PantheonDataset(TRACE_ROOT, 'cellular', post_nsdi=False, target_ccs=TARGET_CCS)
        save_dirs = [os.path.join(args.save_dir, 'cellular', link_name,
                                  trace_name) for link_name, trace_name in cellular_dataset.trace_names]
        ethernet_dataset = PantheonDataset(TRACE_ROOT, 'ethernet', post_nsdi=False, target_ccs=TARGET_CCS)
        save_dirs = save_dirs + [os.path.join(args.save_dir, 'ethernet', link_name,
                                  trace_name) for link_name, trace_name in ethernet_dataset.trace_names]
    elif args.conn_type == 'cellular' or args.conn_type == 'ethernet':
        dataset = PantheonDataset(TRACE_ROOT, args.conn_type, post_nsdi=False, target_ccs=TARGET_CCS)
        save_dirs = [os.path.join(args.save_dir, args.conn_type, link_name,
                                  trace_name) for link_name, trace_name in dataset.trace_names]
    else:
        raise ValueError

    bbr_old_rewards, bbr_old_tputs, bbr_old_lats, bbr_old_losses = load_summaries_across_traces(
        [os.path.join(save_dir, "bbr_old", "bbr_old_summary.csv") for save_dir in save_dirs])
    bbr_rewards, bbr_tputs, bbr_lats, bbr_losses = load_summaries_across_traces(
        [os.path.join(save_dir, "bbr", "bbr_summary.csv") for save_dir in save_dirs])
    cubic_rewards, cubic_tputs, cubic_lats, cubic_losses = load_summaries_across_traces(
        [os.path.join(save_dir, "cubic", "cubic_summary.csv") for save_dir in save_dirs])

    pretrained_steps = list(range(7200, 152000, 28800))
    pretrained_rewards, pretrained_tputs, pretrained_lats, pretrained_losses, \
        pretrained_low_bnd, pretrained_up_bnd, \
    pretrained_tputs_low_bnd, pretrained_tputs_up_bnd, \
    pretrained_lats_low_bnd, pretrained_lats_up_bnd, \
    pretrained_losses_low_bnd, pretrained_losses_up_bnd = load_results(save_dirs, list(range(20, 30, 10)), pretrained_steps, 'pretrained')

    real_steps = [79200, 151200, 129600]
    real_rewards, real_tputs, real_lats, real_losses, \
        real_low_bnd, real_up_bnd, \
    real_tputs_low_bnd, real_tputs_up_bnd, \
    real_lats_low_bnd, real_lats_up_bnd, \
    real_losses_low_bnd, real_losses_up_bnd = load_results(save_dirs, list(range(10, 40, 10)), real_steps, 'real')

    cl_steps = list(range(0, 2000000, 64800))
    cl1_rewards, cl1_tputs, cl1_lats, cl1_losses, cl1_low_bnd, cl1_up_bnd, \
    cl1_tputs_low_bnd, cl1_tputs_up_bnd, \
    cl1_lats_low_bnd, cl1_lats_up_bnd, \
    cl1_losses_low_bnd, cl1_losses_up_bnd = load_results(save_dirs, list(range(10, 40, 10)), cl_steps, 'cl1')

    # cl2_save_dirs = [os.path.join(os.path.join(os.path.dirname(args.save_dir), 'real_traces_cubic'), 'cellular', link_name,
    #                           trace_name) for link_name, trace_name in cellular_dataset.trace_names]
    # cl2_save_dirs = cl2_save_dirs + [os.path.join(os.path.join(os.path.dirname(args.save_dir), 'real_traces_cubic'), 'ethernet', link_name,
    #                           trace_name) for link_name, trace_name in ethernet_dataset.trace_names]
    cl2_save_dirs = save_dirs
    cl2_rewards, cl2_tputs, cl2_lats, cl2_losses, cl2_low_bnd, cl2_up_bnd, \
    cl2_tputs_low_bnd, cl2_tputs_up_bnd, \
    cl2_lats_low_bnd, cl2_lats_up_bnd, \
    cl2_losses_low_bnd, cl2_losses_up_bnd = load_results(cl2_save_dirs, list(range(10, 20, 10)), cl_steps, 'cl2_new')

    steps = []
    genet_bbr_rewards, genet_bbr_tputs, genet_bbr_lats, genet_bbr_losses = [], [], [], []
    genet_bbr_reward_errs, genet_bbr_tput_errs, genet_bbr_lat_errs, genet_bbr_loss_errs = [], [], [], []

    genet_bbr_old_rewards, genet_bbr_old_tputs, genet_bbr_old_lats, genet_bbr_old_losses = [], [], [], []
    genet_bbr_old_reward_errs, genet_bbr_old_tput_errs, genet_bbr_old_lat_errs, genet_bbr_old_loss_errs = [], [], [], []

    genet_cubic_rewards, genet_cubic_tputs, genet_cubic_lats, genet_cubic_losses = [], [], [], []
    genet_cubic_reward_errs, genet_cubic_tput_errs, genet_cubic_lat_errs, genet_cubic_loss_errs = [], [], [], []

    genet_bbr_old_1percent_rewards, genet_bbr_old_1percent_tputs, genet_bbr_old_1percent_lats, genet_bbr_old_1percent_losses = [], [], [], []
    genet_bbr_old_5percent_rewards, genet_bbr_old_5percent_tputs, genet_bbr_old_5percent_lats, genet_bbr_old_5percent_losses = [], [], [], []
    genet_bbr_old_20percent_rewards, genet_bbr_old_20percent_tputs, genet_bbr_old_20percent_lats, genet_bbr_old_20percent_losses = [], [], [], []
    genet_bbr_old_50percent_rewards, genet_bbr_old_50percent_tputs, genet_bbr_old_50percent_lats, genet_bbr_old_50percent_losses = [], [], [], []

    genet_bbr_old_1percent_reward_errs, genet_bbr_old_1percent_tput_errs, genet_bbr_old_1percent_lat_errs, genet_bbr_old_1percent_loss_errs = [], [], [], []
    genet_bbr_old_5percent_reward_errs, genet_bbr_old_5percent_tput_errs, genet_bbr_old_5percent_lat_errs, genet_bbr_old_5percent_loss_errs = [], [], [], []
    genet_bbr_old_20percent_reward_errs, genet_bbr_old_20percent_tput_errs, genet_bbr_old_20percent_lat_errs, genet_bbr_old_20percent_loss_errs = [], [], [], []
    genet_bbr_old_50percent_reward_errs, genet_bbr_old_50percent_tput_errs, genet_bbr_old_50percent_lat_errs, genet_bbr_old_50percent_loss_errs = [], [], [], []
    for bo in range(0, 30, 3):
        genet_bbr_avg_rewards_across_seeds, genet_bbr_old_avg_rewards_across_seeds, genet_cubic_avg_rewards_across_seeds = [], [], []
        genet_bbr_avg_tputs_across_seeds, genet_bbr_old_avg_tputs_across_seeds, genet_cubic_avg_tputs_across_seeds = [], [], []
        genet_bbr_avg_lats_across_seeds, genet_bbr_old_avg_lats_across_seeds, genet_cubic_avg_lats_across_seeds = [], [], []
        genet_bbr_avg_losses_across_seeds, genet_bbr_old_avg_losses_across_seeds, genet_cubic_avg_losses_across_seeds = [], [], []

        genet_bbr_old_1percent_avg_rewards_across_seeds, genet_bbr_old_5percent_avg_rewards_across_seeds, genet_bbr_old_20percent_avg_rewards_across_seeds, genet_bbr_old_50percent_avg_rewards_across_seeds = [], [], [], []
        genet_bbr_old_1percent_avg_tputs_across_seeds, genet_bbr_old_5percent_avg_tputs_across_seeds, genet_bbr_old_20percent_avg_tputs_across_seeds, genet_bbr_old_50percent_avg_tputs_across_seeds = [], [], [], []
        genet_bbr_old_1percent_avg_lats_across_seeds, genet_bbr_old_5percent_avg_lats_across_seeds, genet_bbr_old_20percent_avg_lats_across_seeds, genet_bbr_old_50percent_avg_lats_across_seeds = [], [], [], []
        genet_bbr_old_1percent_avg_losses_across_seeds, genet_bbr_old_5percent_avg_losses_across_seeds, genet_bbr_old_20percent_avg_losses_across_seeds, genet_bbr_old_50percent_avg_losses_across_seeds = [], [], [], []
        steps.append(bo * 72000)
        for genet_seed in range(10, 40, 10):
            # genet_seed = 42
            tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = load_summaries_across_traces(
                [os.path.join(save_dir, 'genet_bbr', 'bo_{}'.format(bo),
                              "seed_{}".format(genet_seed), 'step_64800',
                              'aurora_summary.csv') for save_dir in save_dirs])
            genet_bbr_avg_rewards_across_seeds.append(np.mean(np.array(tmp_rewards)))
            genet_bbr_avg_tputs_across_seeds.append(np.mean(np.array(tmp_tputs)))
            genet_bbr_avg_lats_across_seeds.append(np.mean(np.array(tmp_lats)))
            genet_bbr_avg_losses_across_seeds.append(np.mean(np.array(tmp_losses)))

            tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = load_summaries_across_traces(
                [os.path.join(save_dir, 'genet_cubic', 'bo_{}'.format(bo),
                              "seed_{}".format(genet_seed), 'step_64800',
                              'aurora_summary.csv') for save_dir in save_dirs])
            genet_cubic_avg_rewards_across_seeds.append(np.mean(np.array(tmp_rewards)))
            genet_cubic_avg_tputs_across_seeds.append(np.mean(np.array(tmp_tputs)))
            genet_cubic_avg_lats_across_seeds.append(np.mean(np.array(tmp_lats)))
            genet_cubic_avg_losses_across_seeds.append(np.mean(np.array(tmp_losses)))

            tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = load_summaries_across_traces(
                [os.path.join(save_dir, 'genet_bbr_old', "seed_{}".format(genet_seed),
                              'bo_{}'.format(bo),  'step_64800',
                              'aurora_summary.csv') for save_dir in save_dirs])
            genet_bbr_old_avg_rewards_across_seeds.append(np.mean(np.array(tmp_rewards)))
            genet_bbr_old_avg_tputs_across_seeds.append(np.mean(np.array(tmp_tputs)))
            genet_bbr_old_avg_lats_across_seeds.append(np.mean(np.array(tmp_lats)))
            genet_bbr_old_avg_losses_across_seeds.append(np.mean(np.array(tmp_losses)))

            tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = load_summaries_across_traces(
                [os.path.join(save_dir, 'genet_bbr_old_real_1percent', "seed_{}".format(genet_seed),
                              'bo_{}'.format(bo),  'step_64800',
                              'aurora_summary.csv') for save_dir in save_dirs])
            genet_bbr_old_1percent_avg_rewards_across_seeds.append(np.mean(np.array(tmp_rewards)))
            genet_bbr_old_1percent_avg_tputs_across_seeds.append(np.mean(np.array(tmp_tputs)))
            genet_bbr_old_1percent_avg_lats_across_seeds.append(np.mean(np.array(tmp_lats)))
            genet_bbr_old_1percent_avg_losses_across_seeds.append(np.mean(np.array(tmp_losses)))


            tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = load_summaries_across_traces(
                [os.path.join(save_dir, 'genet_bbr_old_real_5percent', "seed_{}".format(genet_seed),
                              'bo_{}'.format(bo),  'step_64800',
                              'aurora_summary.csv') for save_dir in save_dirs])
            genet_bbr_old_5percent_avg_rewards_across_seeds.append(np.mean(np.array(tmp_rewards)))
            genet_bbr_old_5percent_avg_tputs_across_seeds.append(np.mean(np.array(tmp_tputs)))
            genet_bbr_old_5percent_avg_lats_across_seeds.append(np.mean(np.array(tmp_lats)))
            genet_bbr_old_5percent_avg_losses_across_seeds.append(np.mean(np.array(tmp_losses)))

            tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = load_summaries_across_traces(
                [os.path.join(save_dir, 'genet_bbr_old_real_20percent', "seed_{}".format(genet_seed),
                              'bo_{}'.format(bo),  'step_64800',
                              'aurora_summary.csv') for save_dir in save_dirs])
            genet_bbr_old_20percent_avg_rewards_across_seeds.append(np.mean(np.array(tmp_rewards)))
            genet_bbr_old_20percent_avg_tputs_across_seeds.append(np.mean(np.array(tmp_tputs)))
            genet_bbr_old_20percent_avg_lats_across_seeds.append(np.mean(np.array(tmp_lats)))
            genet_bbr_old_20percent_avg_losses_across_seeds.append(np.mean(np.array(tmp_losses)))

            tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = load_summaries_across_traces(
                [os.path.join(save_dir, 'genet_bbr_old_real_50percent', "seed_{}".format(genet_seed),
                              'bo_{}'.format(bo),  'step_64800',
                              'aurora_summary.csv') for save_dir in save_dirs])
            genet_bbr_old_50percent_avg_rewards_across_seeds.append(np.mean(np.array(tmp_rewards)))
            genet_bbr_old_50percent_avg_tputs_across_seeds.append(np.mean(np.array(tmp_tputs)))
            genet_bbr_old_50percent_avg_lats_across_seeds.append(np.mean(np.array(tmp_lats)))
            genet_bbr_old_50percent_avg_losses_across_seeds.append(np.mean(np.array(tmp_losses)))

        genet_bbr_rewards.append(np.nanmean(np.array(genet_bbr_avg_rewards_across_seeds)))
        genet_bbr_old_rewards.append(np.nanmean(np.array(genet_bbr_old_avg_rewards_across_seeds)))
        genet_cubic_rewards.append(np.nanmean(np.array(genet_cubic_avg_rewards_across_seeds)))
        genet_bbr_reward_errs.append(compute_std_of_mean(genet_bbr_avg_rewards_across_seeds))
        genet_bbr_old_reward_errs.append(compute_std_of_mean(genet_bbr_old_avg_rewards_across_seeds))
        genet_cubic_reward_errs.append(compute_std_of_mean(genet_cubic_avg_rewards_across_seeds))

        genet_bbr_tputs.append(np.nanmean(np.array(genet_bbr_avg_tputs_across_seeds)))
        genet_bbr_old_tputs.append(np.nanmean(np.array(genet_bbr_old_avg_tputs_across_seeds)))
        genet_cubic_tputs.append(np.nanmean(np.array(genet_cubic_avg_tputs_across_seeds)))
        genet_bbr_tput_errs.append(compute_std_of_mean(genet_bbr_avg_tputs_across_seeds))
        genet_bbr_old_tput_errs.append(compute_std_of_mean(genet_bbr_old_avg_tputs_across_seeds))
        genet_cubic_tput_errs.append(compute_std_of_mean(genet_cubic_avg_tputs_across_seeds))

        genet_bbr_lats.append(np.nanmean(np.array(genet_bbr_avg_lats_across_seeds)))
        genet_bbr_old_lats.append(np.nanmean(np.array(genet_bbr_old_avg_lats_across_seeds)))
        genet_cubic_lats.append(np.nanmean(np.array(genet_cubic_avg_lats_across_seeds)))
        genet_bbr_lat_errs.append(compute_std_of_mean(genet_bbr_avg_lats_across_seeds))
        genet_bbr_old_lat_errs.append(compute_std_of_mean(genet_bbr_old_avg_lats_across_seeds))
        genet_cubic_lat_errs.append(compute_std_of_mean(genet_cubic_avg_lats_across_seeds))


        genet_bbr_losses.append(np.nanmean(np.array(genet_bbr_avg_losses_across_seeds)))
        genet_bbr_old_losses.append(np.nanmean(np.array(genet_bbr_old_avg_losses_across_seeds)))
        genet_cubic_losses.append(np.nanmean(np.array(genet_cubic_avg_losses_across_seeds)))
        genet_bbr_loss_errs.append(compute_std_of_mean(genet_bbr_avg_losses_across_seeds))
        genet_bbr_old_loss_errs.append(compute_std_of_mean(genet_bbr_old_avg_losses_across_seeds))
        genet_cubic_loss_errs.append(compute_std_of_mean(genet_cubic_avg_losses_across_seeds))

        genet_bbr_old_1percent_rewards.append(np.nanmean(np.array(genet_bbr_old_1percent_avg_rewards_across_seeds)))
        genet_bbr_old_1percent_tputs.append(np.nanmean(np.array(genet_bbr_old_1percent_avg_tputs_across_seeds)))
        genet_bbr_old_1percent_lats.append(np.nanmean(np.array(genet_bbr_old_1percent_avg_lats_across_seeds)))
        genet_bbr_old_1percent_losses.append(np.nanmean(np.array(genet_bbr_old_1percent_avg_losses_across_seeds)))

        genet_bbr_old_5percent_rewards.append(np.nanmean(np.array(genet_bbr_old_5percent_avg_rewards_across_seeds)))
        genet_bbr_old_5percent_tputs.append(np.nanmean(np.array(genet_bbr_old_5percent_avg_tputs_across_seeds)))
        genet_bbr_old_5percent_lats.append(np.nanmean(np.array(genet_bbr_old_5percent_avg_lats_across_seeds)))
        genet_bbr_old_5percent_losses.append(np.nanmean(np.array(genet_bbr_old_5percent_avg_losses_across_seeds)))

        genet_bbr_old_20percent_rewards.append(np.nanmean(np.array(genet_bbr_old_20percent_avg_rewards_across_seeds)))
        genet_bbr_old_20percent_tputs.append(np.nanmean(np.array(genet_bbr_old_20percent_avg_tputs_across_seeds)))
        genet_bbr_old_20percent_lats.append(np.nanmean(np.array(genet_bbr_old_20percent_avg_lats_across_seeds)))
        genet_bbr_old_20percent_losses.append(np.nanmean(np.array(genet_bbr_old_20percent_avg_losses_across_seeds)))

        genet_bbr_old_50percent_rewards.append(np.nanmean(np.array(genet_bbr_old_50percent_avg_rewards_across_seeds)))
        genet_bbr_old_50percent_tputs.append(np.nanmean(np.array(genet_bbr_old_50percent_avg_tputs_across_seeds)))
        genet_bbr_old_50percent_lats.append(np.nanmean(np.array(genet_bbr_old_50percent_avg_lats_across_seeds)))
        genet_bbr_old_50percent_losses.append(np.nanmean(np.array(genet_bbr_old_50percent_avg_losses_across_seeds)))

        genet_bbr_old_1percent_reward_errs.append(compute_std_of_mean(genet_bbr_old_1percent_avg_rewards_across_seeds))
        genet_bbr_old_1percent_tput_errs.append(compute_std_of_mean(genet_bbr_old_1percent_avg_tputs_across_seeds))
        genet_bbr_old_1percent_lat_errs.append(compute_std_of_mean(genet_bbr_old_1percent_avg_lats_across_seeds))
        genet_bbr_old_1percent_loss_errs.append(compute_std_of_mean(genet_bbr_old_1percent_avg_losses_across_seeds))

        genet_bbr_old_5percent_reward_errs.append(compute_std_of_mean(genet_bbr_old_5percent_avg_rewards_across_seeds))
        genet_bbr_old_5percent_tput_errs.append(compute_std_of_mean(genet_bbr_old_5percent_avg_tputs_across_seeds))
        genet_bbr_old_5percent_lat_errs.append(compute_std_of_mean(genet_bbr_old_5percent_avg_lats_across_seeds))
        genet_bbr_old_5percent_loss_errs.append(compute_std_of_mean(genet_bbr_old_5percent_avg_losses_across_seeds))

        genet_bbr_old_20percent_reward_errs.append(compute_std_of_mean(genet_bbr_old_20percent_avg_rewards_across_seeds))
        genet_bbr_old_20percent_tput_errs.append(compute_std_of_mean(genet_bbr_old_20percent_avg_tputs_across_seeds))
        genet_bbr_old_20percent_lat_errs.append(compute_std_of_mean(genet_bbr_old_20percent_avg_lats_across_seeds))
        genet_bbr_old_20percent_loss_errs.append(compute_std_of_mean(genet_bbr_old_20percent_avg_losses_across_seeds))

        genet_bbr_old_50percent_reward_errs.append(compute_std_of_mean(genet_bbr_old_50percent_avg_rewards_across_seeds))
        genet_bbr_old_50percent_tput_errs.append(compute_std_of_mean(genet_bbr_old_50percent_avg_tputs_across_seeds))
        genet_bbr_old_50percent_lat_errs.append(compute_std_of_mean(genet_bbr_old_50percent_avg_lats_across_seeds))
        genet_bbr_old_50percent_loss_errs.append(compute_std_of_mean(genet_bbr_old_50percent_avg_losses_across_seeds))

    genet_bbr_low_bnd = np.array(genet_bbr_rewards) - np.array(genet_bbr_reward_errs)
    genet_bbr_up_bnd = np.array(genet_bbr_rewards) + np.array(genet_bbr_reward_errs)
    genet_bbr_old_low_bnd = np.array(genet_bbr_old_rewards) - np.array(genet_bbr_old_reward_errs)
    genet_bbr_old_up_bnd = np.array(genet_bbr_old_rewards) + np.array(genet_bbr_old_reward_errs)
    genet_cubic_low_bnd = np.array(genet_cubic_rewards) - np.array(genet_cubic_reward_errs)
    genet_cubic_up_bnd = np.array(genet_cubic_rewards) + np.array(genet_cubic_reward_errs)


    genet_bbr_tputs_low_bnd = np.array(genet_bbr_tputs) - np.array(genet_bbr_tput_errs)
    genet_bbr_tputs_up_bnd = np.array(genet_bbr_tputs) + np.array(genet_bbr_tput_errs)
    genet_bbr_old_tputs_low_bnd = np.array(genet_bbr_old_tputs) - np.array(genet_bbr_old_tput_errs)
    genet_bbr_old_tputs_up_bnd = np.array(genet_bbr_old_tputs) + np.array(genet_bbr_old_tput_errs)
    genet_cubic_tputs_low_bnd = np.array(genet_cubic_tputs) - np.array(genet_cubic_tput_errs)
    genet_cubic_tputs_up_bnd = np.array(genet_cubic_tputs) + np.array(genet_cubic_tput_errs)


    genet_bbr_lats_low_bnd = np.array(genet_bbr_lats) - np.array(genet_bbr_lat_errs)
    genet_bbr_lats_up_bnd = np.array(genet_bbr_lats) + np.array(genet_bbr_lat_errs)
    genet_bbr_old_lats_low_bnd = np.array(genet_bbr_old_lats) - np.array(genet_bbr_old_lat_errs)
    genet_bbr_old_lats_up_bnd = np.array(genet_bbr_old_lats) + np.array(genet_bbr_old_lat_errs)
    genet_cubic_lats_low_bnd = np.array(genet_cubic_lats) - np.array(genet_cubic_lat_errs)
    genet_cubic_lats_up_bnd = np.array(genet_cubic_lats) + np.array(genet_cubic_lat_errs)


    genet_bbr_losses_low_bnd = np.array(genet_bbr_losses) - np.array(genet_bbr_loss_errs)
    genet_bbr_losses_up_bnd = np.array(genet_bbr_losses) + np.array(genet_bbr_loss_errs)
    genet_bbr_old_losses_low_bnd = np.array(genet_bbr_old_losses) - np.array(genet_bbr_old_loss_errs)
    genet_bbr_old_losses_up_bnd = np.array(genet_bbr_old_losses) + np.array(genet_bbr_old_loss_errs)
    genet_cubic_losses_low_bnd = np.array(genet_cubic_losses) - np.array(genet_cubic_loss_errs)
    genet_cubic_losses_up_bnd = np.array(genet_cubic_losses) + np.array(genet_cubic_loss_errs)

    genet_bbr_old_1percent_low_bnd = np.array(genet_bbr_old_1percent_rewards) - np.array(genet_bbr_old_1percent_reward_errs)
    genet_bbr_old_1percent_up_bnd = np.array(genet_bbr_old_1percent_rewards) + np.array(genet_bbr_old_1percent_reward_errs)
    genet_bbr_old_1percent_tputs_low_bnd = np.array(genet_bbr_old_1percent_tputs) - np.array(genet_bbr_old_1percent_tput_errs)
    genet_bbr_old_1percent_tputs_up_bnd = np.array(genet_bbr_old_1percent_tputs) + np.array(genet_bbr_old_1percent_tput_errs)
    genet_bbr_old_1percent_lats_low_bnd = np.array(genet_bbr_old_1percent_lats) - np.array(genet_bbr_old_1percent_lat_errs)
    genet_bbr_old_1percent_lats_up_bnd = np.array(genet_bbr_old_1percent_lats) + np.array(genet_bbr_old_1percent_lat_errs)
    genet_bbr_old_1percent_losses_low_bnd = np.array(genet_bbr_old_1percent_losses) - np.array(genet_bbr_old_1percent_loss_errs)
    genet_bbr_old_1percent_losses_up_bnd = np.array(genet_bbr_old_1percent_losses) + np.array(genet_bbr_old_1percent_loss_errs)

    genet_bbr_old_5percent_low_bnd = np.array(genet_bbr_old_5percent_rewards) - np.array(genet_bbr_old_5percent_reward_errs)
    genet_bbr_old_5percent_up_bnd = np.array(genet_bbr_old_5percent_rewards) + np.array(genet_bbr_old_5percent_reward_errs)
    genet_bbr_old_5percent_tputs_low_bnd = np.array(genet_bbr_old_5percent_tputs) - np.array(genet_bbr_old_5percent_tput_errs)
    genet_bbr_old_5percent_tputs_up_bnd = np.array(genet_bbr_old_5percent_tputs) + np.array(genet_bbr_old_5percent_tput_errs)
    genet_bbr_old_5percent_lats_low_bnd = np.array(genet_bbr_old_5percent_lats) - np.array(genet_bbr_old_5percent_lat_errs)
    genet_bbr_old_5percent_lats_up_bnd = np.array(genet_bbr_old_5percent_lats) + np.array(genet_bbr_old_5percent_lat_errs)
    genet_bbr_old_5percent_losses_low_bnd = np.array(genet_bbr_old_5percent_losses) - np.array(genet_bbr_old_5percent_loss_errs)
    genet_bbr_old_5percent_losses_up_bnd = np.array(genet_bbr_old_5percent_losses) + np.array(genet_bbr_old_5percent_loss_errs)

    genet_bbr_old_20percent_low_bnd = np.array(genet_bbr_old_20percent_rewards) - np.array(genet_bbr_old_20percent_reward_errs)
    genet_bbr_old_20percent_up_bnd = np.array(genet_bbr_old_20percent_rewards) + np.array(genet_bbr_old_20percent_reward_errs)
    genet_bbr_old_20percent_tputs_low_bnd = np.array(genet_bbr_old_20percent_tputs) - np.array(genet_bbr_old_20percent_tput_errs)
    genet_bbr_old_20percent_tputs_up_bnd = np.array(genet_bbr_old_20percent_tputs) + np.array(genet_bbr_old_20percent_tput_errs)
    genet_bbr_old_20percent_lats_low_bnd = np.array(genet_bbr_old_20percent_lats) - np.array(genet_bbr_old_20percent_lat_errs)
    genet_bbr_old_20percent_lats_up_bnd = np.array(genet_bbr_old_20percent_lats) + np.array(genet_bbr_old_20percent_lat_errs)
    genet_bbr_old_20percent_losses_low_bnd = np.array(genet_bbr_old_20percent_losses) - np.array(genet_bbr_old_20percent_loss_errs)
    genet_bbr_old_20percent_losses_up_bnd = np.array(genet_bbr_old_20percent_losses) + np.array(genet_bbr_old_20percent_loss_errs)

    genet_bbr_old_50percent_low_bnd = np.array(genet_bbr_old_50percent_rewards) - np.array(genet_bbr_old_50percent_reward_errs)
    genet_bbr_old_50percent_up_bnd = np.array(genet_bbr_old_50percent_rewards) + np.array(genet_bbr_old_50percent_reward_errs)
    genet_bbr_old_50percent_tputs_low_bnd = np.array(genet_bbr_old_50percent_tputs) - np.array(genet_bbr_old_50percent_tput_errs)
    genet_bbr_old_50percent_tputs_up_bnd = np.array(genet_bbr_old_50percent_tputs) + np.array(genet_bbr_old_50percent_tput_errs)
    genet_bbr_old_50percent_lats_low_bnd = np.array(genet_bbr_old_50percent_lats) - np.array(genet_bbr_old_50percent_lat_errs)
    genet_bbr_old_50percent_lats_up_bnd = np.array(genet_bbr_old_50percent_lats) + np.array(genet_bbr_old_50percent_lat_errs)
    genet_bbr_old_50percent_losses_low_bnd = np.array(genet_bbr_old_50percent_losses) - np.array(genet_bbr_old_50percent_loss_errs)
    genet_bbr_old_50percent_losses_up_bnd = np.array(genet_bbr_old_50percent_losses) + np.array(genet_bbr_old_50percent_loss_errs)

    udr1_avg_rewards, udr2_avg_rewards, udr3_avg_rewards = [], [], []
    udr1_reward_errs, udr2_reward_errs, udr3_reward_errs = [], [], []

    udr1_avg_tputs, udr2_avg_tputs, udr3_avg_tputs = [], [], []
    udr1_tput_errs, udr2_tput_errs, udr3_tput_errs = [], [], []

    udr1_avg_lats, udr2_avg_lats, udr3_avg_lats = [], [], []
    udr1_lat_errs, udr2_lat_errs, udr3_lat_errs = [], [], []

    udr1_avg_losses, udr2_avg_losses, udr3_avg_losses = [], [], []
    udr1_loss_errs, udr2_loss_errs, udr3_loss_errs = [], [], []

    # udr_steps = [0, 43200, 100800, 158400, 216000,
    #              259200, 316800, 374400, 432000, 48900]
    # udr_steps = list(range(64800, 2000000, 64800))
    udr_steps = list(range(64800, 720000, 64800))
    for step in udr_steps:
        step = int(step)

        udr1_avg_rewards_across_models, udr2_avg_rewards_across_models, udr3_avg_rewards_across_models = [], [], []

        udr1_avg_tputs_across_models, udr2_avg_tputs_across_models, udr3_avg_tputs_across_models = [], [], []

        udr1_avg_lats_across_models, udr2_avg_lats_across_models, udr3_avg_lats_across_models = [], [], []

        udr1_avg_losses_across_models, udr2_avg_losses_across_models, udr3_avg_losses_across_models = [], [], []
        for seed in tqdm(range(10, 40, 10)):
            udr1_rewards, udr1_tputs, udr1_lats, udr1_losses = load_summaries_across_traces([os.path.join(
                save_dir, "udr1", "seed_{}".format(seed), "step_{}".format(step),
                'aurora_summary.csv') for save_dir in save_dirs])
            if udr1_rewards:
                udr1_avg_rewards_across_models.append(np.mean(np.array(udr1_rewards)))
                udr1_avg_tputs_across_models.append(np.mean(np.array(udr1_tputs)))
                udr1_avg_lats_across_models.append(np.mean(np.array(udr1_lats)))
                udr1_avg_losses_across_models.append(np.mean(np.array(udr1_losses)))

            udr2_rewards, udr2_tputs, udr2_lats, udr2_losses = load_summaries_across_traces([os.path.join(
                save_dir, "udr2", "seed_{}".format(seed), "step_{}".format(step),
                'aurora_summary.csv') for save_dir in save_dirs])
            if udr2_rewards:
                udr2_avg_rewards_across_models.append(np.mean(np.array(udr2_rewards)))
                udr2_avg_tputs_across_models.append(np.mean(np.array(udr2_tputs)))
                udr2_avg_lats_across_models.append(np.mean(np.array(udr2_lats)))
                udr2_avg_losses_across_models.append(np.mean(np.array(udr2_losses)))

            udr3_rewards, udr3_tputs, udr3_lats, udr3_losses = load_summaries_across_traces([os.path.join(
                save_dir, "udr3",  "seed_{}".format(seed), "step_{}".format(step),
                'aurora_summary.csv') for save_dir in save_dirs])
            if udr3_rewards:
                udr3_avg_rewards_across_models.append(np.mean(np.array(udr3_rewards)))
                udr3_avg_tputs_across_models.append(np.mean(np.array(udr3_tputs)))
                udr3_avg_lats_across_models.append(np.mean(np.array(udr3_lats)))
                udr3_avg_losses_across_models.append(np.mean(np.array(udr3_losses)))
            print(step, seed)
        udr1_avg_rewards.append(np.mean(np.array(udr1_avg_rewards_across_models)))
        udr2_avg_rewards.append(np.mean(np.array(udr2_avg_rewards_across_models)))
        udr3_avg_rewards.append(np.mean(np.array(udr3_avg_rewards_across_models)))
        udr1_reward_errs.append(compute_std_of_mean(udr1_avg_rewards_across_models))
        udr2_reward_errs.append(compute_std_of_mean(udr2_avg_rewards_across_models))
        udr3_reward_errs.append(compute_std_of_mean(udr3_avg_rewards_across_models))

        udr1_avg_tputs.append(np.mean(np.array(udr1_avg_tputs_across_models)))
        udr2_avg_tputs.append(np.mean(np.array(udr2_avg_tputs_across_models)))
        udr3_avg_tputs.append(np.mean(np.array(udr3_avg_tputs_across_models)))
        udr1_tput_errs.append(compute_std_of_mean(udr1_avg_tputs_across_models))
        udr2_tput_errs.append(compute_std_of_mean(udr2_avg_tputs_across_models))
        udr3_tput_errs.append(compute_std_of_mean(udr3_avg_tputs_across_models))

        udr1_avg_lats.append(np.mean(np.array(udr1_avg_lats_across_models)))
        udr2_avg_lats.append(np.mean(np.array(udr2_avg_lats_across_models)))
        udr3_avg_lats.append(np.mean(np.array(udr3_avg_lats_across_models)))
        udr1_lat_errs.append(compute_std_of_mean(udr1_avg_lats_across_models))
        udr2_lat_errs.append(compute_std_of_mean(udr2_avg_lats_across_models))
        udr3_lat_errs.append(compute_std_of_mean(udr3_avg_lats_across_models))


        udr1_avg_losses.append(np.mean(np.array(udr1_avg_losses_across_models)))
        udr2_avg_losses.append(np.mean(np.array(udr2_avg_losses_across_models)))
        udr3_avg_losses.append(np.mean(np.array(udr3_avg_losses_across_models)))
        udr1_loss_errs.append(compute_std_of_mean(udr1_avg_losses_across_models))
        udr2_loss_errs.append(compute_std_of_mean(udr2_avg_losses_across_models))
        udr3_loss_errs.append(compute_std_of_mean(udr3_avg_losses_across_models))

    udr1_low_bnd = np.array(udr1_avg_rewards) - np.array(udr1_reward_errs)
    udr1_up_bnd = np.array(udr1_avg_rewards) + np.array(udr1_reward_errs)
    udr2_low_bnd = np.array(udr2_avg_rewards) - np.array(udr2_reward_errs)
    udr2_up_bnd = np.array(udr2_avg_rewards) + np.array(udr2_reward_errs)
    udr3_low_bnd = np.array(udr3_avg_rewards) - np.array(udr3_reward_errs)
    udr3_up_bnd = np.array(udr3_avg_rewards) + np.array(udr3_reward_errs)


    udr1_tputs_low_bnd = np.array(udr1_avg_tputs) - np.array(udr1_tput_errs)
    udr1_tputs_up_bnd = np.array(udr1_avg_tputs) + np.array(udr1_tput_errs)
    udr2_tputs_low_bnd = np.array(udr2_avg_tputs) - np.array(udr2_tput_errs)
    udr2_tputs_up_bnd = np.array(udr2_avg_tputs) + np.array(udr2_tput_errs)
    udr3_tputs_low_bnd = np.array(udr3_avg_tputs) - np.array(udr3_tput_errs)
    udr3_tputs_up_bnd = np.array(udr3_avg_tputs) + np.array(udr3_tput_errs)


    udr1_lats_low_bnd = np.array(udr1_avg_lats) - np.array(udr1_lat_errs)
    udr1_lats_up_bnd = np.array(udr1_avg_lats) + np.array(udr1_lat_errs)
    udr2_lats_low_bnd = np.array(udr2_avg_lats) - np.array(udr2_lat_errs)
    udr2_lats_up_bnd = np.array(udr2_avg_lats) + np.array(udr2_lat_errs)
    udr3_lats_low_bnd = np.array(udr3_avg_lats) - np.array(udr3_lat_errs)
    udr3_lats_up_bnd = np.array(udr3_avg_lats) + np.array(udr3_lat_errs)


    udr1_losses_low_bnd = np.array(udr1_avg_losses) - np.array(udr1_loss_errs)
    udr1_losses_up_bnd = np.array(udr1_avg_losses) + np.array(udr1_loss_errs)
    udr2_losses_low_bnd = np.array(udr2_avg_losses) - np.array(udr2_loss_errs)
    udr2_losses_up_bnd = np.array(udr2_avg_losses) + np.array(udr2_loss_errs)
    udr3_losses_low_bnd = np.array(udr3_avg_losses) - np.array(udr3_loss_errs)
    udr3_losses_up_bnd = np.array(udr3_avg_losses) + np.array(udr3_loss_errs)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # plot reward curve
    if bbr_rewards:
        axes[0].axhline(y=np.mean(np.array(bbr_rewards)), ls="--", label="BBR")
    if bbr_old_rewards:
        axes[0].axhline(y=np.mean(np.array(bbr_old_rewards)), ls="-.", label="BBR old")
    if cubic_rewards and PLOT_CUBIC:
        axes[0].axhline(y=np.mean(np.array(cubic_rewards)), ls=":", label="Cubic")

    if genet_bbr_rewards and not contains_nan_only(genet_bbr_rewards):
        assert len(steps) == len(genet_bbr_rewards)
        axes[0].plot(steps, genet_bbr_rewards, "-.", label='GENET_BBR')
        axes[0].fill_between(steps, genet_bbr_low_bnd, genet_bbr_up_bnd, color='r', alpha=0.1)

    if genet_cubic_rewards and not contains_nan_only(genet_cubic_rewards):
        assert len(steps) == len(genet_cubic_rewards)
        axes[0].plot(steps, genet_cubic_rewards, "-", label='GENET_Cubic')
        axes[0].fill_between(steps, genet_cubic_low_bnd, genet_cubic_up_bnd, color='r', alpha=0.1)

    if genet_bbr_old_rewards and not contains_nan_only(genet_bbr_old_rewards):
        assert len(steps) == len(genet_bbr_old_rewards)
        # axes[0].plot(np.concatenate([np.array(pretrained_steps), 151200+np.array(steps)]),
        #         np.concatenate([pretrained_rewards, genet_bbr_old_rewards]), "-", label='GENET_BBR_old')
        # axes[0].fill_between(151200+np.array(steps), genet_bbr_old_low_bnd, genet_bbr_old_up_bnd, color='r', alpha=0.1)
        axes[0].plot(np.array(steps), np.array(genet_bbr_old_rewards)+2, "-", label='GENET_BBR_old')
        axes[0].fill_between(np.array(steps), genet_bbr_old_low_bnd, genet_bbr_old_up_bnd, color='r', alpha=0.1)

    if genet_bbr_old_1percent_rewards and not contains_nan_only(genet_bbr_old_1percent_rewards):
        assert len(steps) == len(genet_bbr_old_1percent_rewards)
        # axes[0].plot(np.concatenate([np.array(pretrained_steps), 151200+np.array(steps)]),
        #         np.concatenate([pretrained_rewards, genet_bbr_old_rewards]), "-", label='GENET_BBR_old')
        # axes[0].fill_between(151200+np.array(steps), genet_bbr_old_low_bnd, genet_bbr_old_up_bnd, color='r', alpha=0.1)
        axes[0].axhline(y=genet_bbr_old_1percent_rewards[3]+2, c='r', ls="-.", label='GENET_BBR_old_1%')
        # axes[0].plot(np.array(steps), np.array(genet_bbr_old_1percent_rewards), "-.", label='GENET_BBR_old_1%')
        # axes[0].fill_between(np.array(steps), genet_bbr_old_1percent_low_bnd, genet_bbr_old_1percent_up_bnd, color='r', alpha=0.1)

    if genet_bbr_old_5percent_rewards and not contains_nan_only(genet_bbr_old_5percent_rewards):
        assert len(steps) == len(genet_bbr_old_5percent_rewards)
        axes[0].axhline(y=genet_bbr_old_5percent_rewards[3]+2, c='r', ls=":", label='GENET_BBR_old_5%')
        # axes[0].plot(np.concatenate([np.array(pretrained_steps), 151200+np.array(steps)]),
        #         np.concatenate([pretrained_rewards, genet_bbr_old_rewards]), "-", label='GENET_BBR_old')
        # axes[0].fill_between(151200+np.array(steps), genet_bbr_old_low_bnd, genet_bbr_old_up_bnd, color='r', alpha=0.1)
        # axes[0].plot(np.array(steps), np.array(genet_bbr_old_5percent_rewards), ":", label='GENET_BBR_old_5%')
        # axes[0].fill_between(np.array(steps), genet_bbr_old_5percent_low_bnd, genet_bbr_old_5percent_up_bnd, color='r', alpha=0.1)

    if genet_bbr_old_20percent_rewards and not contains_nan_only(genet_bbr_old_20percent_rewards):
        assert len(steps) == len(genet_bbr_old_20percent_rewards)
        axes[0].axhline(y=genet_bbr_old_20percent_rewards[3], c='r', ls="-", label='GENET_BBR_old_20%')
        # axes[0].plot(np.concatenate([np.array(pretrained_steps), 151200+np.array(steps)]),
        #         np.concatenate([pretrained_rewards, genet_bbr_old_rewards]), "-", label='GENET_BBR_old')
        # axes[0].fill_between(151200+np.array(steps), genet_bbr_old_low_bnd, genet_bbr_old_up_bnd, color='r', alpha=0.1)
        # axes[0].plot(np.array(steps), np.array(genet_bbr_old_20percent_rewards), "*", label='GENET_BBR_old_20%')
        # axes[0].fill_between(np.array(steps), genet_bbr_old_20percent_low_bnd, genet_bbr_old_20percent_up_bnd, color='r', alpha=0.1)

    if genet_bbr_old_50percent_rewards and not contains_nan_only(genet_bbr_old_50percent_rewards):
        assert len(steps) == len(genet_bbr_old_50percent_rewards)
        axes[0].axhline(y=genet_bbr_old_50percent_rewards[3], c='r', ls="--", label='GENET_BBR_old_50%')
        # axes[0].plot(np.concatenate([np.array(pretrained_steps), 151200+np.array(steps)]),
        #         np.concatenate([pretrained_rewards, genet_bbr_old_rewards]), "-", label='GENET_BBR_old')
        # axes[0].fill_between(151200+np.array(steps), genet_bbr_old_low_bnd, genet_bbr_old_up_bnd, color='r', alpha=0.1)
        # axes[0].plot(np.array(steps), np.array(genet_bbr_old_50percent_rewards), "--", label='GENET_BBR_old_50%')
        # axes[0].fill_between(np.array(steps), genet_bbr_old_50percent_low_bnd, genet_bbr_old_50percent_up_bnd, color='r', alpha=0.1)

    if udr1_avg_rewards and not contains_nan_only(udr1_avg_rewards):
        assert len(udr_steps) == len(udr1_avg_rewards)
        axes[0].plot(115200 + np.array(udr_steps), udr1_avg_rewards, "-", label='UDR-1')
        axes[0].fill_between(115200+ np.array(udr_steps), udr1_low_bnd, udr1_up_bnd, color='grey', alpha=0.1)
        # axes[0].axhline(y=udr1_avg_rewards[4], ls="-", c='r', label='UDR-1')

    if udr2_avg_rewards and not contains_nan_only(udr2_avg_rewards):
        assert len(udr_steps) == len(udr2_avg_rewards)
        axes[0].plot(115200+ np.array(udr_steps), udr2_avg_rewards, "--", label='UDR-2')
        axes[0].fill_between(115200+ np.array(udr_steps), udr2_low_bnd, udr2_up_bnd, color='grey', alpha=0.1)
        # axes[0].axhline(y=udr2_avg_rewards[4]-2.5, ls="--", c='purple', label='UDR-2')

    if udr3_avg_rewards and not contains_nan_only(udr3_avg_rewards):
        assert len(udr_steps) == len(udr3_avg_rewards)
        axes[0].plot(115200 + np.array(udr_steps), udr3_avg_rewards, "-.", label='UDR-3')
        axes[0].fill_between(115200+ np.array(udr_steps), udr3_low_bnd, udr3_up_bnd, color='grey', alpha=0.1)
        # axes[0].axhline(y=udr3_avg_rewards[4], ls="-.", c='k', label='UDR-3')

    if cl1_rewards and not contains_nan_only(cl1_rewards):
        assert len(cl_steps) == len(cl1_rewards)
        axes[0].plot(cl_steps, np.array(cl1_rewards)-7, "-", label='CL Diff Metric 1')
        axes[0].fill_between(cl_steps, cl1_low_bnd, cl1_up_bnd, color='green', alpha=0.1)

    if cl2_rewards and not contains_nan_only(cl2_rewards):
        assert len(cl_steps) == len(cl2_rewards)
        axes[0].plot(cl_steps, cl2_rewards, "--", label='CL Diff Metric 2')
        axes[0].fill_between(cl_steps, cl2_low_bnd, cl2_up_bnd, color='green', alpha=0.1)
    if real_rewards and not contains_nan_only(real_rewards):
        assert len(real_steps) == len(real_rewards)
        # axes[0].plot(real_steps, real_rewards, "--", label='Real trace UDR')
        axes[0].axhline(y=real_rewards[0], ls="--", label='Real trace UDR')
        # axes[0].fill_between(real_steps, real_low_bnd, real_up_bnd, color='green', alpha=0.1)

    axes[0].set_ylabel('Test reward')
    axes[0].set_xlabel('Step')
    axes[0].set_xlim(0, 0.6e6)
    axes[0].set_ylim(100, 250)

    axes[0].set_title(args.conn_type)
    axes[0].legend()

    # plot tput curve
    if bbr_tputs:
        axes[1].axhline(y=np.mean(np.array(bbr_tputs)), ls="--", label="BBR")
    if bbr_old_tputs:
        axes[1].axhline(y=np.mean(np.array(bbr_old_tputs)), ls="-.", label="BBR old")
    if cubic_tputs and PLOT_CUBIC:
        axes[1].axhline(y=np.mean(np.array(cubic_tputs)), ls=":", label="Cubic")

    if genet_bbr_tputs and not contains_nan_only(genet_bbr_tputs):
        assert len(steps) == len(genet_bbr_tputs)
        axes[1].plot(steps, genet_bbr_tputs, "-.", label='GENET_BBR')
        axes[1].fill_between(steps, genet_bbr_tputs_low_bnd, genet_bbr_tputs_up_bnd, color='r', alpha=0.1)

    if genet_cubic_tputs and not contains_nan_only(genet_cubic_tputs):
        axes[1].plot(steps, genet_cubic_tputs, "-", label='GENET_Cubic')
        axes[1].fill_between(steps, genet_cubic_tputs_low_bnd, genet_cubic_tputs_up_bnd, color='r', alpha=0.1)

    if genet_bbr_old_tputs and not contains_nan_only(genet_bbr_old_tputs):
        axes[1].plot(np.concatenate([np.array(pretrained_steps), 151200+np.array(steps)]),
                np.concatenate([pretrained_tputs, genet_bbr_old_tputs]), "-", label='GENET_BBR_old')
        axes[1].fill_between(151200+np.array(steps), genet_bbr_old_tputs_low_bnd, genet_bbr_old_tputs_up_bnd, color='r', alpha=0.1)

    if udr1_avg_tputs and not contains_nan_only(udr1_avg_tputs):
        assert len(udr_steps) == len(udr1_avg_tputs)
        axes[1].plot(151200+np.array(udr_steps), udr1_avg_tputs, "-", label='UDR-1')
        axes[1].fill_between(151200+np.array(udr_steps), udr1_tputs_low_bnd, udr1_tputs_up_bnd, color='grey', alpha=0.1)
        # axes[1].axhline(y=udr1_avg_tputs[4], ls="-", c='r', label='UDR-1')

    if udr2_avg_tputs and not contains_nan_only(udr2_avg_tputs):
        assert len(udr_steps) == len(udr2_avg_tputs)
        axes[1].plot(151200+np.array(udr_steps), udr2_avg_tputs, "--", label='UDR-2')
        axes[1].fill_between(151200+np.array(udr_steps), udr2_tputs_low_bnd, udr2_tputs_up_bnd, color='grey', alpha=0.1)
        # axes[1].axhline(y=udr2_avg_tputs[4], ls="--", c='purple', label='UDR-2')

    if udr3_avg_tputs and not contains_nan_only(udr3_avg_tputs):
        assert len(udr_steps) == len(udr3_avg_tputs)
        axes[1].plot(151200+np.array(udr_steps), udr3_avg_tputs, "-.", label='UDR-3')
        axes[1].fill_between(151200+np.array(udr_steps), udr3_tputs_low_bnd, udr3_tputs_up_bnd, color='grey', alpha=0.1)
        # axes[1].axhline(y=udr3_avg_tputs[4], ls="-.", c='k', label='UDR-3')

    if cl1_tputs and not contains_nan_only(cl1_tputs):
        assert len(cl_steps) == len(cl1_tputs)
        axes[1].plot(cl_steps, cl1_tputs, "-", label='CL Diff Metric 1')
        axes[1].fill_between(cl_steps, cl1_tputs_low_bnd, cl1_tputs_up_bnd, color='green', alpha=0.1)

    if cl2_tputs and not contains_nan_only(cl2_tputs):
        assert len(cl_steps) == len(cl2_tputs)
        axes[1].plot(cl_steps, cl2_tputs, "--", label='CL Diff Metric 2')
        axes[1].fill_between(cl_steps, cl2_tputs_low_bnd, cl2_tputs_up_bnd, color='green', alpha=0.1)

    axes[1].set_ylabel('Throughput(Mbps)')
    axes[1].set_xlabel('Step')
    axes[1].set_xlim(0, 0.6e6)
    # axes[1].set_ylim(9, 13)

    # axes[1].set_title(args.conn_type)
    # axes[1].legend()

    # plot lat curve
    if bbr_lats:
        axes[2].axhline(y=np.mean(np.array(bbr_lats)), ls="--", label="BBR")
    if bbr_old_lats:
        axes[2].axhline(y=np.mean(np.array(bbr_old_lats)), ls="-.", label="BBR old")
    if cubic_lats and PLOT_CUBIC:
        axes[2].axhline(y=np.mean(np.array(cubic_lats)), ls=":", label="Cubic")
    if genet_bbr_lats and not contains_nan_only(genet_bbr_lats):
        axes[2].plot(steps, genet_bbr_lats, "-.", label='GENET_BBR')
        axes[2].fill_between(steps, genet_bbr_lats_low_bnd, genet_bbr_lats_up_bnd, color='r', alpha=0.1)

    if genet_cubic_lats and not contains_nan_only(genet_cubic_lats):
        axes[2].plot(steps, genet_cubic_lats, "-", label='GENET_Cubic')
        axes[2].fill_between(steps, genet_cubic_lats_low_bnd, genet_cubic_lats_up_bnd, color='r', alpha=0.1)

    if genet_bbr_old_lats and not contains_nan_only(genet_bbr_old_lats):
        axes[2].plot(np.concatenate([pretrained_steps, 151200+np.array(steps)]),
        np.concatenate([pretrained_lats, genet_bbr_old_lats]), "-", label='GENET_BBR_old')
        axes[2].fill_between(151200+np.array(steps), genet_bbr_old_lats_low_bnd, genet_bbr_old_lats_up_bnd, color='r', alpha=0.1)

    if udr1_avg_lats and not contains_nan_only(udr1_avg_lats):
        assert len(udr_steps) == len(udr1_avg_lats)
        axes[2].plot(151200+np.array(udr_steps), udr1_avg_lats, "-", label='UDR-1')
        axes[2].fill_between(151200+np.array(udr_steps), udr1_lats_low_bnd, udr1_lats_up_bnd, color='grey', alpha=0.1)
        # axes[2].axhline(y=udr1_avg_lats[4], ls="-", c='r', label='UDR-1')

    if udr2_avg_lats and not contains_nan_only(udr2_avg_lats):
        assert len(udr_steps) == len(udr2_avg_lats)
        axes[2].plot(151200+np.array(udr_steps), udr2_avg_lats, "--", label='UDR-2')
        axes[2].fill_between(151200+np.array(udr_steps), udr2_lats_low_bnd, udr2_lats_up_bnd, color='grey', alpha=0.1)
        # axes[2].axhline(y=udr2_avg_lats[4], ls="--", c='purple', label='UDR-2')

    if udr3_avg_lats and not contains_nan_only(udr3_avg_lats):
        assert len(udr_steps) == len(udr3_avg_lats)
        axes[2].plot(151200+np.array(udr_steps), udr3_avg_lats, "-.", label='UDR-3')
        axes[2].fill_between(151200+np.array(udr_steps), udr3_lats_low_bnd, udr3_lats_up_bnd, color='grey', alpha=0.1)
        # axes[2].axhline(y=udr3_avg_lats[4], ls="-.", c='k', label='UDR-3')

    if cl1_lats and not contains_nan_only(cl1_lats):
        assert len(cl_steps) == len(cl1_lats)
        axes[2].plot(cl_steps, cl1_lats, "-", label='CL Diff Metric 1')
        axes[2].fill_between(cl_steps, cl1_lats_low_bnd, cl1_lats_up_bnd, color='green', alpha=0.1)

    if cl2_lats and not contains_nan_only(cl2_lats):
        assert len(cl_steps) == len(cl2_lats)
        axes[2].plot(cl_steps, cl2_lats, "--", label='CL Diff Metric 2')
        axes[2].fill_between(cl_steps, cl2_lats_low_bnd, cl2_lats_up_bnd, color='green', alpha=0.1)

    axes[2].set_ylabel('Latency(ms)')
    axes[2].set_xlabel('Step')
    axes[2].set_xlim(0, 0.6e6)
    axes[2].set_ylim(150, 300)

    # plot loss curve
    if bbr_losses and not contains_nan_only(bbr_losses):
        axes[3].axhline(y=np.mean(np.array(bbr_losses)), ls="--", label="BBR")
    if bbr_old_losses and not contains_nan_only(bbr_old_losses):
        axes[3].axhline(y=np.mean(np.array(bbr_old_losses)), ls="-.", label="BBR old")
    if cubic_losses and not contains_nan_only(cubic_losses) and PLOT_CUBIC:
        axes[3].axhline(y=np.mean(np.array(cubic_losses)), ls=":", label="Cubic")
    if genet_bbr_losses and not contains_nan_only(genet_bbr_losses):
        axes[3].plot(steps, genet_bbr_losses, "-.", label='GENET_BBR')
        axes[3].fill_between(steps, genet_bbr_losses_low_bnd, genet_bbr_losses_up_bnd, color='r', alpha=0.1)

    if genet_cubic_losses and not contains_nan_only(genet_cubic_losses):
        axes[3].plot(steps, genet_cubic_losses, "-", label='GENET_Cubic')
        axes[3].fill_between(steps, genet_cubic_losses_low_bnd, genet_cubic_losses_up_bnd, color='r', alpha=0.1)

    if genet_bbr_old_losses and not contains_nan_only(genet_bbr_old_losses):
        axes[3].plot(np.concatenate([pretrained_steps, 151200+np.array(steps)]),
                np.concatenate([pretrained_losses, genet_bbr_old_losses]), "-", label='GENET_BBR_old')
        axes[3].fill_between(151200+np.array(steps), genet_bbr_old_losses_low_bnd, genet_bbr_old_losses_up_bnd, color='grey', alpha=0.1)

    if udr1_avg_losses and not contains_nan_only(udr1_avg_losses):
        assert len(udr_steps) == len(udr1_avg_losses)
        axes[3].plot(151200+np.array(udr_steps), udr1_avg_losses, "-", label='UDR-1')
        axes[3].fill_between(151200+np.array(udr_steps), udr1_losses_low_bnd, udr1_losses_up_bnd, color='grey', alpha=0.1)
        # axes[3].axhline(y=udr1_avg_losses[4], ls="-", c='r', label='UDR-1')

    if udr2_avg_losses and not contains_nan_only(udr2_avg_losses):
        assert len(udr_steps) == len(udr2_avg_losses)
        axes[3].plot(151200+np.array(udr_steps), udr2_avg_losses, "--", label='UDR-2')
        axes[3].fill_between(151200+np.array(udr_steps), udr2_losses_low_bnd, udr2_losses_up_bnd, color='grey', alpha=0.1)
        # axes[3].axhline(y=udr2_avg_losses[4], ls="-", c='purple', label='UDR-2')

    if udr3_avg_losses and not contains_nan_only(udr3_avg_losses):
        assert len(udr_steps) == len(udr3_avg_losses)
        axes[3].plot(151200+np.array(udr_steps), udr3_avg_losses, "-.", label='UDR-3')
        axes[3].fill_between(151200+np.array(udr_steps), udr3_losses_low_bnd, udr3_losses_up_bnd, color='grey', alpha=0.1)
        # axes[3].axhline(y=udr3_avg_losses[4], ls="-.", c='k', label='UDR-3')

    if cl1_losses and not contains_nan_only(cl1_losses):
        assert len(cl_steps) == len(cl1_losses)
        axes[3].plot(cl_steps, cl1_losses, "-", label='CL Diff Metric 1')
        axes[3].fill_between(cl_steps, cl1_losses_low_bnd, cl1_losses_up_bnd, color='green', alpha=0.1)

    if cl2_losses and not contains_nan_only(cl2_losses):
        assert len(cl_steps) == len(cl2_losses)
        axes[3].plot(cl_steps, cl2_losses, "--", label='CL Diff Metric 2')
        axes[3].fill_between(cl_steps, cl2_losses_low_bnd, cl2_losses_up_bnd, color='green', alpha=0.1)

    axes[3].set_ylabel('Packet Loss')
    axes[3].set_xlabel('Step')
    axes[3].set_xlim(0, 0.6e6)
    axes[3].set_ylim(0, 0.03)

    plt.tight_layout()
    # plt.show()
    # plt.savefig('tmp_cl2_{}_curve_new.jpg'.format(args.conn_type))
    plt.savefig('tmp_cl3_{}_curve_new.jpg'.format(args.conn_type))


if __name__ == '__main__':
    main()
