import argparse
import os
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from common.utils import set_seed
from simulator.aurora import Aurora
from simulator.network_simulator.bbr import BBR
from simulator.network_simulator.cubic import Cubic
from simulator.trace import generate_trace



def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Test latest Aurora model Aurora "
                                     "Testing in simulator.")
    parser.add_argument('--save-dir', type=str, default="",
                        help="direcotry to testing results.")
    parser.add_argument('--log-file', type=str, required=True,
                        help="path to validation log file.")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    # parser.add_argument("--config-file", type=str, default=None,
    #                     help='config file.')

    args, unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    df = pd.read_csv(args.log_file, sep='\t')
    assert isinstance(df, pd.DataFrame)

    latest_step = int(df['num_timesteps'].iloc[-1])
    assert os.path.exists(os.path.join(os.path.dirname(args.log_file),
        "model_step_{}.ckpt.meta".format(latest_step)))
    latest_model_path = os.path.join(os.path.dirname(args.log_file),
                                     "model_step_{}.ckpt".format(latest_step))

    aurora = Aurora(seed=args.seed, timesteps_per_actorbatch=10, log_dir="",
                    pretrained_model_path=latest_model_path)
    bbr = BBR(True)
    cubic = Cubic(True)

    test_traces = []
    trace_dirs = []
    for noise in [0, 20]:
        for bw in [20, 50]:
            tr = generate_trace((30, 30), (bw, bw), (bw, bw), (25, 25), (0, 0), (0.1, 0.1), (60, 60), (noise, noise))
            test_traces.append(tr)

    for _ in range(5):
        test_traces.append(generate_trace((30, 30), (0.1, 0.1), (20, 20), (50, 100), (0, 0), (0.5, 1), (10, 10), (0, 10)))
        test_traces.append(generate_trace((30, 30), (10, 10), (100, 100), (50, 100), (0, 0), (0.5, 1), (10, 10), (0, 10)))

    for i, tr in enumerate(test_traces):
        os.makedirs(os.path.join(args.save_dir, 'trace_{}'.format(i)), exist_ok=True)
        tr.dump(os.path.join(args.save_dir, 'trace_{}'.format(i), 'trace.json'))
        trace_dirs.append(os.path.join(args.save_dir, 'trace_{}'.format(i)))


    t_start = time.time()
    aurora_pkt_level_rewards = []
    for tr, save_dir in zip(test_traces, trace_dirs):
        _, pkt_level_reward = aurora.test(tr, save_dir, True)
        aurora_pkt_level_rewards.append(pkt_level_reward)
    print('aurora', time.time() - t_start)
    t_start = time.time()
    bbr_results = bbr.test_on_traces(test_traces, trace_dirs, True)
    print('bbr', time.time() - t_start)
    t_start = time.time()
    cubic_results = cubic.test_on_traces(test_traces, trace_dirs, True)
    print('cubic', time.time() - t_start)

    bbr_pkt_level_rewards = [val for _, val in bbr_results]
    cubic_pkt_level_rewards = [val for _, val in cubic_results]
    mean_rewards = [np.mean(aurora_pkt_level_rewards),
                    np.mean(bbr_pkt_level_rewards),
                    np.mean(cubic_pkt_level_rewards)]
    reward_errs = [np.std(aurora_pkt_level_rewards),
                   np.std(bbr_pkt_level_rewards),
                   np.std(cubic_pkt_level_rewards)]
    plt.bar([1, 2, 3], mean_rewards, yerr=reward_errs, width=0.5)
    plt.xticks([1, 2, 3], ['aurora', 'bbr', 'cubic'])
    plt.ylabel('Test Reward')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'test_cc.jpg'))


if __name__ == "__main__":
    main()
