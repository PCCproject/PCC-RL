import argparse
import os
from typing import Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from simulator.network_simulator.constants import BITS_PER_BYTE, BYTES_PER_PACKET
from simulator.trace import Trace
from common.utils import pcc_aurora_reward


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Plot time series figures.")
    parser.add_argument('--log-file', type=str, nargs="+", required=True,
                        help="path to a testing log file.")
    parser.add_argument('--trace-file', type=str, default=None,
                        help="path to a trace file.")
    parser.add_argument('--save-dir', type=str, default=None,
                        help="path to save.")
    parser.add_argument('--noise', type=float, default=0)

    args, unknown = parser.parse_known_args()
    return args


def plot(trace: Union[Trace, None], log_file: str, save_dir: str, cc: str):
    df = pd.read_csv(log_file)
    fig, axes = plt.subplots(6, 1, figsize=(12, 10))
    axes[0].set_title(cc)
    axes[0].plot(df['timestamp'], df['recv_rate'] / 1e6, 'o-', ms=2,
            label='throughput, avg {:.3f}mbps'.format(
                     df['recv_rate'].mean() / 1e6))
    axes[0].plot(df['timestamp'], df['send_rate'] / 1e6, 'o-', ms=2,
            label='send rate, avg {:.3f}mbps'.format(
                     df['send_rate'].mean() / 1e6))

    if trace:
        avg_bw = trace.avg_bw
        min_rtt = trace.min_delay * 2 / 1e3
        axes[0].plot(trace.timestamps, trace.bandwidths, 'o-', ms=2, drawstyle='steps-post',
                     label='bw, avg {:.3f}mbps'.format(avg_bw))
    else:
        axes[0].plot(df['timestamp'], df['bandwidth'] / 1e6,
                     label='bw, avg {:.3f}mbps'.format(df['bandwidth'].mean() / 1e6))
        avg_bw = df['bandwidth'].mean() / 1e6
        min_rtt = None
    axes[0].set_xlabel("Time(s)")
    axes[0].set_ylabel("mbps")
    axes[0].legend(loc='right')
    axes[0].set_ylim(0, )
    axes[0].set_xlim(0, )

    axes[1].plot(df['timestamp'], df['latency']*1000,
            label='RTT avg {:.3f}ms'.format(df['latency'].mean()*1000))
    axes[1].set_xlabel("Time(s)")
    axes[1].set_ylabel("Latency(ms)")
    axes[1].legend(loc='right')
    axes[1].set_xlim(0, )
    axes[1].set_ylim(0, )

    axes[2].plot(df['timestamp'], df['loss'],
            label='loss avg {:.3f}'.format(df['loss'].mean()))
    axes[2].set_xlabel("Time(s)")
    axes[2].set_ylabel("loss")
    axes[2].legend()
    axes[2].set_xlim(0, )
    axes[2].set_ylim(0, 1)

    avg_reward_mi = pcc_aurora_reward(
            df['recv_rate'].mean() / BITS_PER_BYTE / BYTES_PER_PACKET,
            df['latency'].mean(), df['loss'].mean(),
            avg_bw * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET, min_rtt)


    axes[3].plot(df['timestamp'], df['reward'],
            label='rewards avg {:.3f}'.format(avg_reward_mi))
    axes[3].set_xlabel("Time(s)")
    axes[3].set_ylabel("Reward")
    axes[3].legend()
    axes[3].set_xlim(0, )
    # axes[3].set_ylim(, )

    axes[4].plot(df['timestamp'], df['action'] * 1.0,
                 label='delta avg {:.3f}'.format(df['action'].mean()))
    axes[4].set_xlabel("Time(s)")
    axes[4].set_ylabel("delta")
    axes[4].legend()
    axes[4].set_xlim(0, )

    axes[5].plot(df['timestamp'], df['packet_in_queue'] /
                 df['queue_size'], label='Queue Occupancy')
    axes[5].set_xlabel("Time(s)")
    axes[5].set_ylabel("Queue occupancy")
    axes[5].legend()
    axes[5].set_xlim(0, )
    axes[5].set_ylim(0, 1)

    # axes[5].plot(df['timestamp'], df['cwnd'], label='cwnd')
    # axes[5].plot(df['timestamp'], df['ssthresh'], label='ssthresh')
    # axes[5].set_xlabel("Time(s)")
    # axes[5].set_ylabel("# packets")
    # axes[5].set_ylim(0, df['cwnd'].max())
    # axes[5].legend()
    # axes[5].set_xlim(0, )
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "{}_time_series.jpg".format(cc)))
    plt.close()

    # if log_idx == 0:
    #     print("{},{},{},{},{},".format(os.path.dirname(log_file), df['recv_rate'].mean()/1e6,
    #                                    df['latency'].mean()*1000,
    #                                    df['loss'].mean(),
    #                                    df['reward'].mean()), end='')
    # else:
    #     print("{},{},{},{},".format(df['recv_rate'].mean()/1e6,
    #                                 df['latency'].mean()*1000,
    #                                 df['loss'].mean(),
    #                                 df['reward'].mean()), end='')
    # if log_idx + 1 == len(args.log_file) and args.trace_file and args.trace_file.endswith('.log'):
    #     trace = Trace.load_from_pantheon_file(args.trace_file, 50, 0, 50)
    #     bw_avg = np.mean(trace.bandwidths)
    #     bw_std = np.std(trace.bandwidths)
    #     rtt_avg = np.mean(trace.delays)
    #     T_s_bw, change_cnt = compute_T_s_bw(trace.timestamps, trace.bandwidths)
    #     bw_range = max(trace.bandwidths) - min(trace.bandwidths)
    #
    #     print("{},{},{},{},{},{}".format(bw_avg, bw_std, T_s_bw, change_cnt, bw_range, rtt_avg))


def main():
    args = parse_args()
    for _, log_file in enumerate(args.log_file):
        if not os.path.exists(log_file):
            continue
        if not args.trace_file:
            trace = None
        elif args.trace_file.endswith('.json'):
            trace = Trace.load_from_file(args.trace_file)
        elif args.trace_file.endswith('.log'):
            trace = Trace.load_from_pantheon_file(args.trace_file, loss=0, queue=10)
        else:
            trace = None
        cc = os.path.basename(log_file).split('_')[0]
        plot(trace, log_file, args.save_dir, cc)


if __name__ == "__main__":
    main()
