import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from common.utils import pcc_aurora_reward


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Plot packet log figures.")
    parser.add_argument('--log-file', type=str, nargs="+", required=True,
                        help="path to a testing log file.")
    parser.add_argument('--save-dir', type=str, default=None,
                        help="path to save.")

    return parser.parse_args()


class PacketLog():
    def __init__(self, packet_log_file, ms_bin_size=500):
        self.pkt_log_file = packet_log_file
        self.pkt_sent_ts = []
        # # pkt_sent_bytes = []
        self.pkt_acked_ts = []
        self.pkt_rtt = []
        self.pkt_queue_delays = []
        self.bin_size = ms_bin_size / 1000
        self.first_ts = None

        self.binwise_bytes_sent = {}
        self.binwise_bytes_acked = {}
        self.binwise_bytes_lost = {}
        with open(packet_log_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                ts = float(line[0])
                pkt_id = int(line[1])
                pkt_type = line[2]
                pkt_byte = int(line[3])
                if self.first_ts is None:
                    self.first_ts = ts
                if pkt_type == 'acked':
                    rtt = float(line[4]) * 1000
                    queue_delay = float(line[5]) * 1000
                    self.pkt_acked_ts.append(ts)
                    self.pkt_rtt.append(rtt)
                    self.pkt_queue_delays.append(queue_delay)

                    bin_id = self.ts_to_bin_id(ts)
                    self.binwise_bytes_acked[bin_id] = self.binwise_bytes_acked.get(
                        bin_id, 0) + pkt_byte
                elif pkt_type == 'sent':
                    self.pkt_sent_ts.append(ts)
                    bin_id = self.ts_to_bin_id(ts)
                    self.binwise_bytes_sent[bin_id] = self.binwise_bytes_sent.get(
                        bin_id, 0) + pkt_byte
                elif pkt_type == 'lost':
                    bin_id = self.ts_to_bin_id(ts)
                    self.binwise_bytes_lost[bin_id] = self.binwise_bytes_lost.get(
                        bin_id, 0) + pkt_byte
                # elif pkt_type == 'lost':
                # print(line.slrip('\n'))

        # return pkt_sent_ts, pkt_acked_ts, pkt_rtt, pkt_queue_delays
    def ts_to_bin_id(self, ts):
        return int((ts - self.first_ts) / self.bin_size)

    def get_throughput(self):
        throughput_ts = []
        throughput = []
        for bin_id in sorted(self.binwise_bytes_acked):
            throughput_ts.append((bin_id + 1) * self.bin_size)
            throughput.append(
                self.binwise_bytes_acked[bin_id] * 8 / self.bin_size / 1e6)
        return throughput_ts, throughput

    def get_sending_rate(self):
        sending_rate_ts = []
        sending_rate = []
        for bin_id in sorted(self.binwise_bytes_sent):
            sending_rate_ts.append((bin_id + 1) * self.bin_size)
            sending_rate.append(
                self.binwise_bytes_sent[bin_id] * 8 / self.bin_size / 1e6)
        return sending_rate_ts, sending_rate

    def get_rtt(self):
        return self.pkt_acked_ts, self.pkt_rtt

    def get_queue_delay(self):
        return self.pkt_acked_ts, self.pkt_queue_delays

    def get_loss_rate(self):
        return 1 - len(self.pkt_acked_ts) / len(self.pkt_sent_ts)


def main():
    args = parse_args()

    for log_file in args.log_file:
        pkt_log = PacketLog(log_file)
        cc = os.path.splitext(os.path.basename(log_file))[0].split('_')[0]

        fig, axes = plt.subplots(2, 1, figsize=(6, 8))
        sending_rate_ts, sending_rate = pkt_log.get_sending_rate()
        throughput_ts, throughput = pkt_log.get_throughput()
        rtt_ts, rtt = pkt_log.get_rtt()
        queue_delay_ts, queue_delay = pkt_log.get_queue_delay()
        loss = pkt_log.get_loss_rate()
        reward = pcc_aurora_reward(
            np.mean(throughput) * 1e6 / 8 / 1500, np.mean(rtt) / 1e3, loss)
        print(throughput[:10])
        axes[0].plot(throughput_ts, throughput,
                     label='throughput, avg {:.3f}Mbps'.format(np.mean(throughput)))
        axes[0].plot(sending_rate_ts, sending_rate,
                     label='sending rate, avg {:.3f}Mbps'.format(np.mean(sending_rate)))
        axes[0].legend()
        axes[0].set_xlabel("Time(s)")
        axes[0].set_ylabel("Rate(Mbps)")
        axes[0].set_title('{} loss rate = {:.3f}, reward = {:3f}'.format(
            cc, loss, reward))

        axes[1].plot(
            rtt_ts, rtt, label='RTT, avg {:.3f}ms'.format(np.mean(rtt)))
        # axes[1].plot(queue_delay_ts, queue_delay, label='Queue delay, avg {:.3f}ms'.format(np.mean(queue_delay)))
        axes[1].legend()
        axes[1].set_xlabel("Time(s)")
        axes[1].set_ylabel("Latency(ms)")
        axes[1].set_title('{} loss rate = {:.3f}, reward = {:3f}'.format(
            cc, loss, reward))

        plt.tight_layout()
        if args.save_dir:
            plt.savefig(os.path.join(args.save_dir,
                        'binwise_{}_plot.png'.format(cc)))


if __name__ == '__main__':
    main()

# plt.show()
