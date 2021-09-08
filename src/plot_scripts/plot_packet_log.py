import argparse
import csv
import os
from typing import Dict, List, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common.utils import pcc_aurora_reward
from simulator.network_simulator.constants import BITS_PER_BYTE, BYTES_PER_PACKET
from simulator.trace import Trace


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Plot packet log figures.")
    parser.add_argument('--log-file', type=str, nargs="+", required=True,
                        help="path to a testing log file.")
    parser.add_argument('--save-dir', type=str, default=None,
                        help="path to save.")
    parser.add_argument('--trace-file', type=str, default=None,
                        help="path to trace file.")

    return parser.parse_args()


class PacketLog():
    def __init__(self, pkt_sent_ts: List[float], pkt_acked_ts: List[float],
                 pkt_rtt: List[float], pkt_queue_delays: List[float],
                 first_ts, binwise_bytes_sent: Dict[int, int],
                 binwise_bytes_acked: Dict[int, int],
                 binwise_bytes_lost: Dict[int, int], packet_log_file=None, ms_bin_size: int = 500):
        self.pkt_log_file = packet_log_file
        self.pkt_sent_ts = pkt_sent_ts
        self.pkt_acked_ts = pkt_acked_ts
        self.pkt_rtt = pkt_rtt
        self.pkt_queue_delays = pkt_queue_delays
        self.bin_size = ms_bin_size / 1000
        self.first_ts = first_ts

        self.binwise_bytes_sent = binwise_bytes_sent
        self.binwise_bytes_acked = binwise_bytes_acked
        self.binwise_bytes_lost = binwise_bytes_lost

        self.avg_sending_rate = None
        self.avg_throughput = None
        self.avg_latency = None

    @classmethod
    def from_log_file(cls, packet_log_file: str, ms_bin_size: int = 500):
        pkt_sent_ts = []
        pkt_acked_ts = []
        pkt_rtt = []
        pkt_queue_delays = []
        bin_size = ms_bin_size / 1000
        first_ts = None

        binwise_bytes_sent = {}
        binwise_bytes_acked = {}
        binwise_bytes_lost = {}
        with open(packet_log_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if line[0] == 'timestamp':
                    continue
                ts = float(line[0])
                pkt_id = int(line[1])
                pkt_type = line[2]
                pkt_byte = int(line[3])
                if first_ts is None:
                    first_ts = ts
                # if ts - first_ts < 2:
                #     continue
                if pkt_type == 'acked':
                    rtt = float(line[4]) * 1000
                    queue_delay = float(line[5]) * 1000
                    pkt_acked_ts.append(ts)
                    pkt_rtt.append(rtt)
                    pkt_queue_delays.append(queue_delay)

                    bin_id = cls.ts_to_bin_id(ts, first_ts, bin_size)
                    binwise_bytes_acked[bin_id] = binwise_bytes_acked.get(
                        bin_id, 0) + pkt_byte
                elif pkt_type == 'sent':
                    pkt_sent_ts.append(ts)
                    bin_id = cls.ts_to_bin_id(ts, first_ts, bin_size)
                    binwise_bytes_sent[bin_id] = binwise_bytes_sent.get(
                        bin_id, 0) + pkt_byte
                elif pkt_type == 'lost':
                    bin_id = cls.ts_to_bin_id(ts, first_ts, bin_size)
                    binwise_bytes_lost[bin_id] = binwise_bytes_lost.get(
                        bin_id, 0) + pkt_byte
                elif pkt_type == 'arrived':
                    pass
                else:
                    raise RuntimeError(
                        "Unrecognized pkt_type {}!".format(pkt_type))
        return cls(pkt_sent_ts, pkt_acked_ts, pkt_rtt, pkt_queue_delays,
                   first_ts, binwise_bytes_sent, binwise_bytes_acked,
                   binwise_bytes_lost, packet_log_file=packet_log_file,
                   ms_bin_size=ms_bin_size)

    @classmethod
    def from_log(cls, pkt_log, ms_bin_size: int = 500):
        pkt_sent_ts = []
        pkt_acked_ts = []
        pkt_rtt = []
        pkt_queue_delays = []
        bin_size = ms_bin_size / 1000
        first_ts = None

        binwise_bytes_sent = {}
        binwise_bytes_acked = {}
        binwise_bytes_lost = {}
        first_ts = None
        for line in pkt_log:
            ts = line[0]
            pkt_id = line[1]
            pkt_type = line[2]
            pkt_byte = line[3]
            if first_ts is None:
                first_ts = ts
            if pkt_type == 'acked':
                rtt = float(line[4]) * 1000
                queue_delay = float(line[5]) * 1000
                pkt_acked_ts.append(ts)
                pkt_rtt.append(rtt)
                pkt_queue_delays.append(queue_delay)

                bin_id = cls.ts_to_bin_id(ts, first_ts, bin_size)
                binwise_bytes_acked[bin_id] = binwise_bytes_acked.get(
                    bin_id, 0) + pkt_byte
            elif pkt_type == 'sent':
                pkt_sent_ts.append(ts)
                bin_id = cls.ts_to_bin_id(ts, first_ts, bin_size)
                binwise_bytes_sent[bin_id] = binwise_bytes_sent.get(
                    bin_id, 0) + pkt_byte
            elif pkt_type == 'lost':
                bin_id = cls.ts_to_bin_id(ts, first_ts, bin_size)
                binwise_bytes_lost[bin_id] = binwise_bytes_lost.get(
                    bin_id, 0) + pkt_byte
            elif pkt_type == 'arrived':
                pass
            else:
                raise RuntimeError(
                    "Unrecognized pkt_type {}!".format(pkt_type))
        return cls(pkt_sent_ts, pkt_acked_ts, pkt_rtt, pkt_queue_delays,
                   first_ts, binwise_bytes_sent, binwise_bytes_acked,
                   binwise_bytes_lost, packet_log_file=None,
                   ms_bin_size=ms_bin_size)

    @staticmethod
    def ts_to_bin_id(ts, first_ts, bin_size) -> int:
        return int((ts - first_ts) / bin_size)

    @staticmethod
    def bin_id_to_s(bin_id, bin_size) -> int:
        return bin_id * bin_size

    def get_throughput(self) -> Tuple[List[float], List[float]]:
        throughput_ts = []
        throughput = []
        for bin_id in sorted(self.binwise_bytes_acked):
            throughput_ts.append(self.bin_id_to_s(bin_id, self.bin_size))
            throughput.append(
                self.binwise_bytes_acked[bin_id] * BITS_PER_BYTE / self.bin_size / 1e6)
        return throughput_ts, throughput

    def get_sending_rate(self) -> Tuple[List[float], List[float]]:
        sending_rate_ts = []
        sending_rate = []
        for bin_id in sorted(self.binwise_bytes_sent):
            sending_rate_ts.append(self.bin_id_to_s(bin_id, self.bin_size))
            sending_rate.append(
                self.binwise_bytes_sent[bin_id] * BITS_PER_BYTE / self.bin_size / 1e6)
        return sending_rate_ts, sending_rate

    def get_rtt(self) -> Tuple[List[float], List[float]]:
        return self.pkt_acked_ts, self.pkt_rtt

    def get_queue_delay(self) -> Tuple[List[float], List[float]]:
        return self.pkt_acked_ts, self.pkt_queue_delays

    def get_loss_rate(self) -> float:
        return 1 - len(self.pkt_acked_ts) / len(self.pkt_sent_ts)

    def get_reward(self, trace_file: str, trace=None) -> float:
        if trace_file and trace_file.endswith('.json'):
            trace = Trace.load_from_file(trace_file)
        elif trace_file and trace_file.endswith('.log'):
            trace = Trace.load_from_pantheon_file(trace_file, 0, 50, 500)
        loss = self.get_loss_rate()
        if trace is None:
            # original reward
            return pcc_aurora_reward(
                self.get_avg_throughput() * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET,
                self.get_avg_latency() / 1e3, loss)
        # normalized reward
        return pcc_aurora_reward(
            self.get_avg_throughput() * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET,
            self.get_avg_latency() / 1e3, loss,
            trace.avg_bw * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET,
            trace.min_delay * 2 / 1e3)

    def get_avg_sending_rate(self) -> float:
        if not self.pkt_sent_ts:
            return 0.0
        if self.avg_sending_rate is None:
            total_duration = self.pkt_sent_ts[-1] - self.pkt_sent_ts[0]
            bytes_sum = 0
            for _, bytes_sent in self.binwise_bytes_sent.items():
                bytes_sum += bytes_sent
            self.avg_sending_rate = bytes_sum * BITS_PER_BYTE / 1e6 / total_duration
        return self.avg_sending_rate

    def get_avg_throughput(self) -> float:
        if not self.pkt_acked_ts:
            return 0.0
        if self.avg_throughput is None:
            total_duration = self.pkt_acked_ts[-1] - self.pkt_acked_ts[0]
            bytes_sum = 0
            for _, bytes_acked in self.binwise_bytes_acked.items():
                bytes_sum += bytes_acked
            self.avg_throughput = bytes_sum * BITS_PER_BYTE / 1e6 / total_duration
        return self.avg_throughput

    def get_avg_latency(self) -> float:
        if self.avg_latency is None:
            self.avg_latency = np.mean(self.pkt_rtt)
        return self.avg_latency


def plot(trace: Union[Trace, None], pkt_log: PacketLog, save_dir: str, cc: str):
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    sending_rate_ts, sending_rate = pkt_log.get_sending_rate()
    throughput_ts, throughput = pkt_log.get_throughput()
    rtt_ts, rtt = pkt_log.get_rtt()
    queue_delay_ts, queue_delay = pkt_log.get_queue_delay()
    loss = pkt_log.get_loss_rate()
    # print(throughput[:10])
    axes[0].plot(throughput_ts, throughput, "-o", ms=2,  # drawstyle='steps-post',
                 label='throughput, avg {:.3f}Mbps'.format(pkt_log.get_avg_throughput()))
    axes[0].plot(sending_rate_ts, sending_rate, "-o", ms=2,  # drawstyle='steps-post',
                 label='sending rate, avg {:.3f}Mbps'.format(pkt_log.get_avg_sending_rate()))
    if trace is not None:
        axes[0].plot(trace.timestamps, trace.bandwidths, "-o", ms=2,  # drawstyle='steps-post',
                     label='bandwidth, avg {:.3f}Mbps'.format(np.mean(trace.bandwidths)))
        queue_size = trace.queue_size
        trace_random_loss = trace.loss_rate
        delay_noise = trace.delay_noise
    else:
        queue_size = "N/A"
        trace_random_loss = "N/A"
        delay_noise = "N/A"
        # axes[0].plot(np.arange(30), np.ones_like(np.arange(30)) * 6, "-o", ms=2,  # drawstyle='steps-post',
        #              label='bandwidth, avg {:.3f}Mbps'.format(6))
    axes[0].legend()
    axes[0].set_xlabel("Time(s)")
    axes[0].set_ylabel("Rate(Mbps)")
    axes[0].set_xlim(0, )
    axes[0].set_ylim(0, )
    reward = pkt_log.get_reward("", None)
    normalized_reward = reward
    if trace is not None:
        normalized_reward = pkt_log.get_reward("", trace)
    axes[0].set_title('{} reward={:.3f}, normalized reward={:.3f}'.format(
        cc, reward, normalized_reward))

    axes[1].plot(rtt_ts, rtt, ms=2, label='RTT, avg {:.3f}ms'.format(
        pkt_log.get_avg_latency()))
    # axes[1].plot(queue_delay_ts, queue_delay, label='Queue delay, avg {:.3f}ms'.format(np.mean(queue_delay)))
    if trace is not None:
        axes[1].plot(rtt_ts, np.ones_like(rtt) * 2 * trace.min_delay, c='C2',
                     label="trace minRTT {:.3f}ms".format(2*min(trace.delays)))
    axes[1].legend()
    axes[1].set_xlabel("Time(s)")
    axes[1].set_ylabel("Latency(ms)")
    axes[1].set_title('{} loss={:.3f}, rand loss={:.3f}, queue={}, lat_noise={:.3f}'.format(
        cc, loss, trace_random_loss, queue_size, delay_noise))
    axes[1].set_xlim(0, )
    # axes[1].set_ylim(0, )

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'binwise_{}_plot.png'.format(cc)))
    plt.close()

def main():
    args = parse_args()
    if args.trace_file and args.trace_file.endswith('.json'):
        trace = Trace.load_from_file(args.trace_file)
    elif args.trace_file and args.trace_file.endswith('.log'):
        trace = Trace.load_from_pantheon_file(
            args.trace_file, 0, 50, 500)
    else:
        trace = None

    for log_idx, log_file in enumerate(args.log_file):
        if not os.path.exists(log_file):
            continue
        pkt_log = PacketLog.from_log_file(log_file, 500)
        cc = os.path.splitext(os.path.basename(log_file))[0].split('_')[0]
        plot(trace, pkt_log, args.save_dir, cc)


        # if log_idx == 0:
        #     print("{},{},{},{},{},".format(os.path.dirname(log_file),
        #                                    np.mean(throughput), np.mean(rtt), loss, reward), end=',')
        # else:
        #     print("{},{},{},{},".format(np.mean(throughput),
        #                                 np.mean(rtt), loss, reward), end=',')



if __name__ == '__main__':
    main()
