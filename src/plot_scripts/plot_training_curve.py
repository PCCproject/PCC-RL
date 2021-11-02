import argparse
import glob
import os
from typing import List, Tuple

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from common.utils import compute_std_of_mean, load_summary
from simulator.trace import Trace

TRACE_ROOT = "../../data"
# TRACE_ROOT = "../../data/cellular/2018-12-11T00-27-AWS-Brazil-2-to-Colombia-cellular-3-runs-3-flows"

TARGET_CCS = ["bbr", "cubic", "vegas", "indigo", "ledbat", "quic"]


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Plot the performance curves on real "
                                     "Pantheon traces over training.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to testing results.")
    parser.add_argument("--conn-type", type=str, required=True,
                        choices=('cellular', 'ethernet', 'wifi'),
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
            print(log_file, 'does not exist')
            continue
        summary = load_summary(log_file)
        rewards.append(summary['pkt_level_reward'])
        tputs.append(summary['average_throughput'])
        lats.append(summary['average_latency'])
        losses.append(summary['loss_rate'])
    return rewards, tputs, lats, losses


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
        for cc in TARGET_CCS:
            print("Loading {}, {} traces collected by {}...".format(args.conn_type, link_name, cc))
            for trace_file in tqdm(sorted(glob.glob(os.path.join(link_dir, "{}_datalink_run[1-3].log".format(cc))))):
                traces.append(Trace.load_from_pantheon_file(
                    trace_file, 0.0, queue_size, front_offset=0))
                save_dir = os.path.join(args.save_dir, args.conn_type, link_name,
                                        os.path.splitext(os.path.basename(trace_file))[0])
                save_dirs.append(save_dir)

    bbr_old_rewards, bbr_old_tputs, bbr_old_lats, bbr_old_losses = load_summaries_across_traces(
        [os.path.join(save_dir, "bbr_old", "bbr_old_summary.csv") for save_dir in save_dirs])
    bbr_rewards, bbr_tputs, bbr_lats, bbr_losses = load_summaries_across_traces(
        [os.path.join(save_dir, "bbr", "bbr_summary.csv") for save_dir in save_dirs])
    cubic_rewards, cubic_tputs, cubic_lats, cubic_losses = load_summaries_across_traces(
        [os.path.join(save_dir, "cubic", "cubic_summary.csv") for save_dir in save_dirs])
    steps = []
    genet_bbr_rewards, genet_bbr_tputs, genet_bbr_lats, genet_bbr_losses = [], [], [], []
    genet_bbr_old_rewards, genet_bbr_old_tputs, genet_bbr_old_lats, genet_bbr_old_losses = [], [], [], []
    genet_cubic_rewards, genet_cubic_tputs, genet_cubic_lats, genet_cubic_losses = [], [], [], []
    for bo in range(0, 30, 3):
        genet_seed = 42
        tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = load_summaries_across_traces(
            [os.path.join(save_dir, 'genet_bbr', 'bo_{}'.format(bo),
                          "seed_{}".format(genet_seed), 'step_64800',
                          'aurora_summary.csv') for save_dir in save_dirs])
        # [os.path.join(save_dir, 'genet_bbr', 'bo_{}'.format(bo), 'step_64800', 'aurora_summary.csv') for save_dir in save_dirs])
        steps.append(bo * 72000)
        genet_bbr_rewards.append(np.mean(tmp_rewards))
        genet_bbr_tputs.append(np.mean(tmp_tputs))
        genet_bbr_lats.append(np.mean(tmp_lats))
        genet_bbr_losses.append(np.mean(tmp_losses))
        tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = load_summaries_across_traces(
            [os.path.join(save_dir, 'genet_cubic', 'bo_{}'.format(bo),
                          "seed_{}".format(genet_seed), 'step_64800',
                          'aurora_summary.csv') for save_dir in save_dirs])
        # [os.path.join(save_dir, 'genet_cubic', 'bo_{}'.format(bo), 'step_64800','aurora_summary.csv') for save_dir in save_dirs])
        genet_cubic_rewards.append(np.mean(tmp_rewards))
        genet_cubic_tputs.append(np.mean(tmp_tputs))
        genet_cubic_lats.append(np.mean(tmp_lats))
        genet_cubic_losses.append(np.mean(tmp_losses))

        tmp_rewards, tmp_tputs, tmp_lats, tmp_losses = load_summaries_across_traces(
            [os.path.join(save_dir, 'genet_bbr_old', "seed_{}".format(genet_seed),
                          'bo_{}'.format(bo),  'step_64800',
                          'aurora_summary.csv') for save_dir in save_dirs])
        # [os.path.join(save_dir, 'genet_bbr', 'bo_{}'.format(bo), 'step_64800', 'aurora_summary.csv') for save_dir in save_dirs])
        genet_bbr_old_rewards.append(np.mean(tmp_rewards))
        genet_bbr_old_tputs.append(np.mean(tmp_tputs))
        genet_bbr_old_lats.append(np.mean(tmp_lats))
        genet_bbr_old_losses.append(np.mean(tmp_losses))

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
    udr_steps = list(range(64800, 2000000, 64800))
    for step in udr_steps:
        step = int(step)

        udr1_avg_rewards_across_models, udr2_avg_rewards_across_models, udr3_avg_rewards_across_models = [], [], []

        udr1_avg_tputs_across_models, udr2_avg_tputs_across_models, udr3_avg_tputs_across_models = [], [], []

        udr1_avg_lats_across_models, udr2_avg_lats_across_models, udr3_avg_lats_across_models = [], [], []

        udr1_avg_losses_across_models, udr2_avg_losses_across_models, udr3_avg_losses_across_models = [], [], []
        for seed in tqdm(range(10, 110, 10)):
            udr1_rewards, udr1_tputs, udr1_lats, udr1_losses = load_summaries_across_traces([os.path.join(
                save_dir, "udr1", "seed_{}".format(seed), "step_{}".format(step),
                'aurora_summary.csv') for save_dir in save_dirs])
            if udr1_rewards:
                udr1_avg_rewards_across_models.append(np.mean(udr1_rewards))
                udr1_avg_tputs_across_models.append(np.mean(udr1_tputs))
                udr1_avg_lats_across_models.append(np.mean(udr1_lats))
                udr1_avg_losses_across_models.append(np.mean(udr1_losses))

            udr2_rewards, udr2_tputs, udr2_lats, udr2_losses = load_summaries_across_traces([os.path.join(
                save_dir, "udr2", "seed_{}".format(seed), "step_{}".format(step),
                'aurora_summary.csv') for save_dir in save_dirs])
            if udr2_rewards:
                udr2_avg_rewards_across_models.append(np.mean(udr2_rewards))
                udr2_avg_tputs_across_models.append(np.mean(udr2_tputs))
                udr2_avg_lats_across_models.append(np.mean(udr2_lats))
                udr2_avg_losses_across_models.append(np.mean(udr2_losses))

            udr3_rewards, udr3_tputs, udr3_lats, udr3_losses = load_summaries_across_traces([os.path.join(
                save_dir, "udr3",  "seed_{}".format(seed), "step_{}".format(step),
                'aurora_summary.csv') for save_dir in save_dirs])
            if udr3_rewards:
                udr3_avg_rewards_across_models.append(np.mean(udr3_rewards))
                udr3_avg_tputs_across_models.append(np.mean(udr3_tputs))
                udr3_avg_lats_across_models.append(np.mean(udr3_lats))
                udr3_avg_losses_across_models.append(np.mean(udr3_losses))
        udr1_avg_rewards.append(np.mean(udr1_avg_rewards_across_models))
        udr2_avg_rewards.append(np.mean(udr2_avg_rewards_across_models))
        udr3_avg_rewards.append(np.mean(udr3_avg_rewards_across_models))
        udr1_reward_errs.append(compute_std_of_mean(udr1_avg_rewards_across_models))
        udr2_reward_errs.append(compute_std_of_mean(udr2_avg_rewards_across_models))
        udr3_reward_errs.append(compute_std_of_mean(udr3_avg_rewards_across_models))

        udr1_avg_tputs.append(np.mean(udr1_avg_tputs_across_models))
        udr2_avg_tputs.append(np.mean(udr2_avg_tputs_across_models))
        udr3_avg_tputs.append(np.mean(udr3_avg_tputs_across_models))
        udr1_tput_errs.append(compute_std_of_mean(udr1_avg_tputs_across_models))
        udr2_tput_errs.append(compute_std_of_mean(udr2_avg_tputs_across_models))
        udr3_tput_errs.append(compute_std_of_mean(udr3_avg_tputs_across_models))

        udr1_avg_lats.append(np.mean(udr1_avg_lats_across_models))
        udr2_avg_lats.append(np.mean(udr2_avg_lats_across_models))
        udr3_avg_lats.append(np.mean(udr3_avg_lats_across_models))
        udr1_lat_errs.append(compute_std_of_mean(udr1_avg_lats_across_models))
        udr2_lat_errs.append(compute_std_of_mean(udr2_avg_lats_across_models))
        udr3_lat_errs.append(compute_std_of_mean(udr3_avg_lats_across_models))


        udr1_avg_losses.append(np.mean(udr1_avg_losses_across_models))
        udr2_avg_losses.append(np.mean(udr2_avg_losses_across_models))
        udr3_avg_losses.append(np.mean(udr3_avg_losses_across_models))
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
    axes[0].axhline(y=np.mean(bbr_rewards), ls="--", label="BBR")
    axes[0].axhline(y=np.mean(bbr_old_rewards), ls="-.", label="BBR old")
    axes[0].axhline(y=np.mean(cubic_rewards), ls=":", label="Cubic")
    axes[0].plot(steps, genet_bbr_rewards, "-.", label='GENET_BBR')
    axes[0].plot(steps, genet_cubic_rewards, "-", label='GENET_Cubic')
    axes[0].plot(steps, genet_bbr_old_rewards, "-", label='GENET_BBR_old')

    assert len(udr_steps) == len(udr1_avg_rewards)
    axes[0].plot(udr_steps, udr1_avg_rewards, "-", label='UDR-1')
    axes[0].fill_between(udr_steps, udr1_low_bnd, udr1_up_bnd, color='grey', alpha=0.1)

    assert len(udr_steps) == len(udr2_avg_rewards)
    axes[0].plot(udr_steps, udr2_avg_rewards, "--", label='UDR-2')
    axes[0].fill_between(udr_steps, udr2_low_bnd, udr2_up_bnd, color='grey', alpha=0.1)

    assert len(udr_steps) == len(udr3_avg_rewards)
    axes[0].plot(udr_steps, udr3_avg_rewards, "-.", label='UDR-3')
    axes[0].fill_between(udr_steps, udr3_low_bnd, udr3_up_bnd, color='grey', alpha=0.1)

    axes[0].set_ylabel('Test reward')
    axes[0].set_xlabel('Step')

    axes[0].set_title(args.conn_type)
    axes[0].legend()

    # plot tput curve
    axes[1].axhline(y=np.mean(bbr_tputs), ls="--", label="BBR")
    axes[1].axhline(y=np.mean(bbr_old_tputs), ls="-.", label="BBR old")
    axes[1].axhline(y=np.mean(cubic_tputs), ls=":", label="Cubic")
    axes[1].plot(steps, genet_bbr_tputs, "-.", label='GENET_BBR')
    axes[1].plot(steps, genet_cubic_tputs, "-", label='GENET_Cubic')
    axes[1].plot(steps, genet_bbr_old_tputs, "-", label='GENET_BBR_old')

    assert len(udr_steps) == len(udr1_avg_tputs)
    axes[1].plot(udr_steps, udr1_avg_tputs, "-", label='UDR-1')
    axes[1].fill_between(udr_steps, udr1_tputs_low_bnd, udr1_tputs_up_bnd, color='grey', alpha=0.1)

    assert len(udr_steps) == len(udr2_avg_tputs)
    axes[1].plot(udr_steps, udr2_avg_tputs, "--", label='UDR-2')
    axes[1].fill_between(udr_steps, udr2_tputs_low_bnd, udr2_tputs_up_bnd, color='grey', alpha=0.1)

    assert len(udr_steps) == len(udr3_avg_tputs)
    axes[1].plot(udr_steps, udr3_avg_tputs, "-.", label='UDR-3')
    axes[1].fill_between(udr_steps, udr3_tputs_low_bnd, udr3_tputs_up_bnd, color='grey', alpha=0.1)

    axes[1].set_ylabel('Throughput(Mbps)')
    axes[1].set_xlabel('Step')

    axes[1].set_title(args.conn_type)
    axes[1].legend()

    # plot lat curve
    axes[2].axhline(y=np.mean(bbr_lats), ls="--", label="BBR")
    axes[2].axhline(y=np.mean(bbr_old_lats), ls="-.", label="BBR old")
    axes[2].axhline(y=np.mean(cubic_lats), ls=":", label="Cubic")
    axes[2].plot(steps, genet_bbr_lats, "-.", label='GENET_BBR')
    axes[2].plot(steps, genet_cubic_lats, "-", label='GENET_Cubic')
    axes[2].plot(steps, genet_bbr_old_lats, "-", label='GENET_BBR_old')

    assert len(udr_steps) == len(udr1_avg_lats)
    axes[2].plot(udr_steps, udr1_avg_lats, "-", label='UDR-1')
    axes[2].fill_between(udr_steps, udr1_lats_low_bnd, udr1_lats_up_bnd, color='grey', alpha=0.1)

    assert len(udr_steps) == len(udr2_avg_lats)
    axes[2].plot(udr_steps, udr2_avg_lats, "--", label='UDR-2')
    axes[2].fill_between(udr_steps, udr2_lats_low_bnd, udr2_lats_up_bnd, color='grey', alpha=0.1)

    assert len(udr_steps) == len(udr3_avg_lats)
    axes[2].plot(udr_steps, udr3_avg_lats, "-.", label='UDR-3')
    axes[2].fill_between(udr_steps, udr3_lats_low_bnd, udr3_lats_up_bnd, color='grey', alpha=0.1)

    axes[2].set_ylabel('Latency(ms)')
    axes[2].set_xlabel('Step')

    axes[2].set_title(args.conn_type)
    axes[2].legend()

    # plot loss curve
    axes[3].axhline(y=np.mean(bbr_losses), ls="--", label="BBR")
    axes[3].axhline(y=np.mean(bbr_old_losses), ls="-.", label="BBR old")
    axes[3].axhline(y=np.mean(cubic_losses), ls=":", label="Cubic")
    axes[3].plot(steps, genet_bbr_losses, "-.", label='GENET_BBR')
    axes[3].plot(steps, genet_cubic_losses, "-", label='GENET_Cubic')
    axes[3].plot(steps, genet_bbr_old_losses, "-", label='GENET_BBR_old')

    assert len(udr_steps) == len(udr1_avg_losses)
    axes[3].plot(udr_steps, udr1_avg_losses, "-", label='UDR-1')
    axes[3].fill_between(udr_steps, udr1_losses_low_bnd, udr1_losses_up_bnd, color='grey', alpha=0.1)

    assert len(udr_steps) == len(udr2_avg_losses)
    axes[3].plot(udr_steps, udr2_avg_losses, "--", label='UDR-2')
    axes[3].fill_between(udr_steps, udr2_losses_low_bnd, udr2_losses_up_bnd, color='grey', alpha=0.1)

    assert len(udr_steps) == len(udr3_avg_losses)
    axes[3].plot(udr_steps, udr3_avg_losses, "-.", label='UDR-3')
    axes[3].fill_between(udr_steps, udr3_losses_low_bnd, udr3_losses_up_bnd, color='grey', alpha=0.1)

    axes[3].set_ylabel('Packet Loss')
    axes[3].set_xlabel('Step')

    axes[3].set_title(args.conn_type)
    axes[3].legend()

    plt.tight_layout()
    plt.show()
    # plt.savefig('train_curve.jpg')


if __name__ == '__main__':
    main()
