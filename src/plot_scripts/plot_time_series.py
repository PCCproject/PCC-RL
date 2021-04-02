import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import ipdb
import matplotlib
matplotlib.use('Agg')


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Plot time series figures.")
    parser.add_argument('--log-file', type=str, nargs="+", required=True,
                        help="path to a testing log file.")
    parser.add_argument('--save-dir', type=str, default=None,
                        help="path to save.")

    return parser.parse_args()


def main():
    args = parse_args()
    print(args.log_file)
    for log_file in args.log_file:
        df = pd.read_csv(log_file)
        fig, axes = plt.subplots(5, 1, figsize=(10, 10))

        axes[0].plot(df['timestamp'], df['throughput'] * 1500 * 8 / 1000000,
                     label='throughput, avg {:.3f}mbps'.format(df['throughput'].mean() * 1500 * 8 / 1000000))
        axes[0].plot(df['timestamp'], df['send_throughput'] * 1500 * 8 / 1000000,
                     label='send rate, avg {:.3f}mbps'.format(df['send_throughput'].mean() * 1500 * 8 / 1000000))
        axes[0].plot(df['timestamp'], df['link0_bw'] * 1500 * 8 / 1000000,
                     label='bw, avg {:.3f}mbps'.format(df['link0_bw'].mean() * 1500 * 8 / 1000000))
        # axes[1].plot(np.arange(0, 11, 0.1), 2.0 * np.sin(2*np.arange(0, 11, 0.1)) + 2,
        #              label='bw, avg {:.3f}mbps'.format(df['link0_bw'].mean() * 1500 * 8 / 1000000))
        axes[0].set_xlabel("Time(s)")
        axes[0].set_ylabel("mbps")
        axes[0].legend()
        axes[0].set_ylim(0, 6)
        axes[0].set_xlim(0, )

        axes[1].plot(df['timestamp'], df['latency']*1000,
                     label='RTT avg {:.3f}ms'.format(df['latency'].mean()*1000))
        axes[1].plot(df['timestamp'], df['queue_delay']*1000,
                     label='Queue Delay avg {:.3f}ms'.format(df['queue_delay'].mean() * 1000))
        axes[1].set_xlabel("Time(s)")
        axes[1].set_ylabel("Latency(ms)")
        axes[1].legend()
        axes[1].set_xlim(0, )
        axes[1].set_ylim(0, 80)


        axes[2].plot(df['timestamp'], df['loss'], label='loss avg {:.3f}'.format(df['loss'].mean()))
        axes[2].set_xlabel("Time(s)")
        axes[2].set_ylabel("loss")
        axes[2].legend()
        axes[2].set_xlim(0, )
        axes[2].set_ylim(0, 1)

        axes[3].plot(df['timestamp'], df['reward'],
                     label='rewards avg {:.3f}'.format(df['reward'].mean()))
        axes[3].set_xlabel("Time(s)")
        axes[3].set_ylabel("Reward")
        axes[3].legend()
        axes[3].set_xlim(0, )

        axes[4].plot(df['timestamp'], df['action'] * 1.0, label='delta')
        axes[4].set_xlabel("Time(s)")
        axes[4].set_ylabel("delta")
        axes[4].legend()
        axes[4].set_xlim(0, )


        # axes[5].plot(df['timestamp'], df['cwnd'], label='cwnd')
        # axes[5].plot(df['timestamp'], df['ssthresh'], label='ssthresh')
        # axes[5].set_xlabel("Time(s)")
        # axes[5].set_ylabel("# packets")
        # axes[5].set_ylim(0, df['cwnd'].max())
        # axes[5].legend()
        # axes[5].set_xlim(0, )
        print(df['reward'].mean())
        plt.tight_layout()
        if args.save_dir is not None:
            if 'cubic' in log_file:
                plt.savefig(os.path.join(
                    args.save_dir, "cubic_time_series.png"))
            elif 'aurora' in log_file:
                plt.savefig(os.path.join(
                    args.save_dir, "aurora_time_series.png"))
    # plt.show()


if __name__ == "__main__":
    main()
