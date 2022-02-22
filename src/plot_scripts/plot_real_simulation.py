import argparse
import csv
import os
import numpy as np

from common.utils import load_summary
from simulator.pantheon_dataset import PantheonDataset

TRACE_ROOT = '../../data'
MODEL_TRAINED_ON_REAL_TRACES = '/datamirror/zxxia/PCC-RL/results_1006/train_on_real_trace_default/real_traces'
# GENET = '/datamirror/zxxia/results_1006/genet_against_real_trace_latest_default/real_traces'
GENET = '/datamirror/zxxia/results_1006/genet_against_real_trace_latest_default_better_reward/real_traces'
# GENET = '/datamirror/zxxia/results_1006/genet_better_bw/real_traces'


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Summary perf on real traces.")
    parser.add_argument('--root', type=str, required=True,
                        help="path to a testing log file.")
    parser.add_argument("--conn-type", type=str, required=True,
                        choices=('ethernet', 'cellular', 'wifi'),
                        help='connection type')
    parser.add_argument('--cc', type=str, required=True,
                        choices=("bbr", 'bbr_old', "cubic", "udr1", "udr2",
                                 "udr3", "genet_bbr", 'genet_bbr_old',
                                 'genet_cubic', 'real'),
                        help='congestion control name')
    parser.add_argument('--save-dir', type=str, required=True,
                        help="path to save summary file.")
    parser.add_argument('--seed', type=int, required=True,
                        help='seed used in training.')
    parser.add_argument('--bo', type=int, default=None,
                        help='seed used in training.')

    return parser.parse_args()

def main():
    args = parse_args()

    if args.cc == 'real':
        summary_file = os.path.join(args.save_dir, "summary_seed_{}.csv".format(args.seed))
    elif args.cc in ('genet_bbr_old', 'genet_bbr', 'genet_cubic'):
        if args.bo is None:
            raise ValueError
        summary_file = os.path.join(args.save_dir, "summary_seed_{}_bo_{}.csv".format(args.seed, args.bo))
    else:
        raise ValueError

    with open(summary_file, 'w') as f:
        writer = csv.DictWriter(
            f, ['link_name', 'run_name', 'trace_avg_bw', 'trace_avg_lat',
                'avg_tput', 'avg_lat', 'loss', 'reward'], lineterminator='\n')
        writer.writeheader()
        dataset = PantheonDataset(TRACE_ROOT, args.conn_type)
        rewards = []
        trace_bw = []
        trace_lat = []
        tput = []
        lat = []
        loss = []
        for link_name, run_name in dataset.trace_names:
            if args.cc == 'real':
                summary_path = os.path.join(
                    args.root, args.conn_type, link_name, run_name,
                    args.cc, 'seed_{}'.format(args.seed), 'aurora_summary.csv')
            elif args.cc in ('genet_bbr_old', 'genet_bbr', 'genet_cubic'):
                if args.bo is None:
                    raise ValueError
                summary_path = os.path.join(
                    args.root, args.conn_type, link_name, run_name,
                    args.cc, 'seed_{}'.format(args.seed), 'bo_{}'.format(args.bo), "step_64800", 'aurora_summary.csv')
            else:
                raise ValueError
            if os.path.exists(summary_path):
                summary = load_summary(summary_path)
                trace_bw.append(summary['trace_average_bandwidth'])
                trace_lat.append(summary['trace_average_latency'])
                tput.append(summary['average_throughput'])
                lat.append(summary['average_latency'])
                loss.append(summary['loss_rate'])
                rewards.append(summary['pkt_level_reward'])
                row = {'link_name': link_name, 'run_name': run_name,
                       'trace_avg_bw': summary['trace_average_bandwidth'],
                       'trace_avg_lat': summary['trace_average_latency'],
                       'avg_tput': summary['average_throughput'],
                       'avg_lat': summary['average_latency'],
                       'loss': summary['loss_rate'],
                       'reward': summary['pkt_level_reward']}
                writer.writerow(row)
        row = {'link_name': "Average", 'run_name': "",
               'trace_avg_bw': np.mean(np.array(trace_bw)),
               'trace_avg_lat': np.mean(np.array(trace_lat)),
               'avg_tput': np.mean(np.array(tput)),
               'avg_lat': np.mean(np.array(lat)),
               'loss': np.mean(np.array(loss)),
               'reward': np.mean(np.array(rewards))}
        writer.writerow(row)


if __name__ == "__main__":
    main()
