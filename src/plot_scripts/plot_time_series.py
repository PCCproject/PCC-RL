import argparse
import ipdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Plot time series figures.")
    parser.add_argument('--log-file', type=str, required=True,
                        help="path to a testing log file.")

    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.log_file)
    fig, axes = plt.subplots(4, 1, figsize=(11, 8))
    axes[0].plot(df['timestamp'], df['cwnd'], label='cwnd')
    axes[0].plot(df['timestamp'], df['ssthresh'], label='ssthresh')
    axes[0].set_xlabel("Time(s)")
    axes[0].set_ylabel("# packets")
    axes[0].set_ylim(0, df['cwnd'].max())
    axes[0].legend()

    axes[1].plot(df['timestamp'], df['throughput'] * 1500 * 8 /1000000, label='throughput, avg {}mbps'.format(df['throughput'].mean() * 1500 * 8 /1000000))
    axes[1].plot(df['timestamp'], df['send_throughput'] * 1500 * 8 /1000000, label='send rate, avg {}mbps'.format(df['send_throughput'].mean() * 1500 * 8 /1000000))
    axes[1].plot(df['timestamp'], df['link0_bw'] * 1500 * 8 /1000000, label='bw, avg {}mbps'.format(df['link0_bw'].mean() * 1500 * 8 /1000000))
    axes[1].set_xlabel("Time(s)")
    # axes[1].set_ylabel("packets/sec")
    axes[1].set_ylabel("mbps")
    axes[1].legend()

    axes[2].plot(df['timestamp'], df['latency']*1000, label='latency')
    axes[2].set_xlabel("Time(s)")
    axes[2].set_ylabel("Latency(ms)")
    axes[2].legend()

    axes[3].plot(df['timestamp'], df['reward'], label='rewards')
    axes[3].set_xlabel("Time(s)")
    axes[3].set_ylabel("Reward")
    axes[3].legend()

    print(df['reward'].mean())
    plt.tight_layout()
    plt.figure()
    plt.plot(df['timestamp'], df['throughput'] * 1500 * 8 /1000000, 'x-', label='throughput, avg {:.2f}mbps'.format(df['throughput'].mean() * 1500 * 8 /1000000))
    plt.plot(df['timestamp'], df['send_throughput'] * 1500 * 8 /1000000, 'x-', label='send rate, avg {:.2f}mbps'.format(df['send_throughput'].mean() * 1500 * 8 /1000000))
    plt.plot(df['timestamp'], df['link0_bw'] * 1500 * 8 /1000000, 'x-', label='bw, avg {:.2f}mbps'.format(df['link0_bw'].mean() * 1500 * 8 /1000000))
    plt.title("PCC Aurora Simulator")
    plt.xlabel("Time(s)")
    plt.ylabel("packets/sec")
    plt.ylabel("mbps")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
