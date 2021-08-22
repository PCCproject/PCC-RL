import argparse
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulator.network_simulator.constants import BITS_PER_BYTE, BYTES_PER_PACKET
from simulator.trace import Trace
from common.utils import pcc_aurora_reward


PLOT = True


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


def main():
    args = parse_args()
    # print(args.log_file)
    for log_idx, log_file in enumerate(args.log_file):
        if not os.path.exists(log_file):
            continue
        cc = os.path.basename(log_file).split('_')[0]
        df = pd.read_csv(log_file)
        if PLOT:
            fig, axes = plt.subplots(6, 1, figsize=(12, 10))
            axes[0].set_title(cc)
            avg_send_rate = df['bytes_sent'].sum() / df['timestamp'].iloc[-1] * BITS_PER_BYTE /1e6
            avg_recv_rate = df['bytes_acked'].sum() / df['timestamp'].iloc[-1] * BITS_PER_BYTE /1e6
            axes[0].plot(df['timestamp'], df['recv_rate'] / 1e6, 'o-', ms=2,
                    label='throughput, avg {:.3f}mbps, {:.3f}'.format(
                             df['recv_rate'].mean() / 1e6, avg_recv_rate))
            axes[0].plot(df['timestamp'], df['send_rate'] / 1e6, 'o-', ms=2,
                    label='send rate, avg {:.3f}mbps, {:.3f}'.format(
                             df['send_rate'].mean() / 1e6, avg_send_rate))

            if args.trace_file:
                if args.trace_file.endswith('.json'):
                    trace = Trace.load_from_file(args.trace_file)
                elif args.trace_file.endswith('.log'):
                    trace = Trace.load_from_pantheon_file(args.trace_file, delay=5, loss=0, queue=10)
                else:
                    raise RuntimeError
                avg_bw = np.mean(trace.bandwidths)
                min_rtt = np.mean(trace.delays) * 2 / 1e3
                axes[0].plot(trace.timestamps, trace.bandwidths, 'o-', ms=2, drawstyle='steps-post',
                             label='bw, avg {:.3f}mbps'.format(avg_bw))
                axes[0].plot(np.arange(0, trace.timestamps[-1], 0.01),
                             [trace.get_bandwidth(ts) for ts in np.arange(0, trace.timestamps[-1], 0.01)],
                             label='bw, avg {:.3f}mbps'.format(df['bandwidth'].mean() / 1e6))
            else:
                axes[0].plot(df['timestamp'], df['bandwidth'] / 1e6,
                             label='bw, avg {:.3f}mbps'.format(df['bandwidth'].mean() / 1e6))
                avg_bw = df['bandwidth'].mean() / 1e6
            if args.noise != 0:
                axes[0].set_title('noise ~ max(0, N(0.004, {})'.format(args.noise))
            axes[0].set_xlabel("Time(s)")
            axes[0].set_ylabel("mbps")
            axes[0].legend(loc='right')
            axes[0].set_ylim(0, )
            axes[0].set_xlim(0, )


            avg_lat = (df['latency'] * df['bytes_acked']).sum() / df['bytes_acked'].sum() * 1000
            axes[1].plot(df['timestamp'], df['latency']*1000,
                    label='RTT avg {:.3f}ms, {:.3f}'.format(df['latency'].mean()*1000, avg_lat))
            # axes[1].plot(df['timestamp'], df['queue_delay']*1000,
            #              label='Queue Delay avg {:.3f}ms'.format(df['queue_delay'].mean() * 1000))
            axes[1].set_xlabel("Time(s)")
            axes[1].set_ylabel("Latency(ms)")
            axes[1].legend(loc='right')
            axes[1].set_xlim(0, )
            axes[1].set_ylim(0, )


            avg_loss = df['bytes_lost'].sum()/df['bytes_sent'].sum()
            axes[2].plot(df['timestamp'], df['loss'],
                    label='loss avg {:.3f}, {:.3f}'.format(
                        df['loss'].mean(), avg_loss))
            axes[2].set_xlabel("Time(s)")
            axes[2].set_ylabel("loss")
            axes[2].legend()
            axes[2].set_xlim(0, )
            axes[2].set_ylim(0, 1)

            avg_reward = pcc_aurora_reward(avg_recv_rate / 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET,
                                           avg_lat /1000, avg_loss,
                                           avg_bw / 1e6/ BITS_PER_BYTE / BYTES_PER_PACKET, min_rtt)

            avg_reward_mi = pcc_aurora_reward(
                    df['recv_rate'].mean() / BITS_PER_BYTE / BYTES_PER_PACKET,
                    df['latency'].mean(), df['loss'].mean(),
                    avg_bw * 1e6/ BITS_PER_BYTE / BYTES_PER_PACKET, min_rtt)


            axes[3].plot(df['timestamp'], df['reward'],
                    label='rewards avg {:.3f}, {:.3f}'.format(avg_reward_mi, avg_reward))
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
            if args.save_dir is not None:
                plt.savefig(os.path.join(args.save_dir,
                                         "{}_time_series.png".format(cc)))

        if log_idx == 0:
            print("{},{},{},{},{},".format(os.path.dirname(log_file), df['recv_rate'].mean()/1e6,
                                           df['latency'].mean()*1000,
                                           df['loss'].mean(),
                                           df['reward'].mean()), end='')
        else:
            print("{},{},{},{},".format(df['recv_rate'].mean()/1e6,
                                        df['latency'].mean()*1000,
                                        df['loss'].mean(),
                                        df['reward'].mean()), end='')
        if log_idx + 1 == len(args.log_file) and args.trace_file and args.trace_file.endswith('.log'):
            trace = Trace.load_from_pantheon_file(args.trace_file, 50, 0, 50)
            bw_avg = np.mean(trace.bandwidths)
            bw_std = np.std(trace.bandwidths)
            rtt_avg = np.mean(trace.delays)
            T_s_bw, change_cnt = compute_T_s_bw(trace.timestamps, trace.bandwidths)
            bw_range = max(trace.bandwidths) - min(trace.bandwidths)

            print("{},{},{},{},{},{}".format(bw_avg, bw_std, T_s_bw, change_cnt, bw_range, rtt_avg))


def compute_T_s_bw(tstamps, bws):
    print(bws, file=sys.stderr)
    prev_ts = tstamps[0]
    prev_bw = bws[0]
    t_s_bw = []
    for ts, bw in zip(tstamps[1:], bws[1:]):
        if np.abs(bw - prev_bw) / prev_bw >= 0.5:
            t_s_bw.append(ts - prev_ts)
            prev_ts = ts
            prev_bw = bw
    if not t_s_bw:
        return 30, 0  # trace duration
    return np.mean(t_s_bw), len(t_s_bw) / 30


if __name__ == "__main__":
    main()
