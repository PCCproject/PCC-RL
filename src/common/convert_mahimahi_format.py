import argparse
import os
import glob

import numpy as np
from simulator.good_network_sim import BYTES_PER_PACKET

from simulator.trace import Trace

IN_FILE = './tmp/'
OUT_FILE = './mahimahi/'
FILE_SIZE = 0
BYTES_PER_PKT = 1500
MILLISEC_IN_SEC = 1000
EXP_LEN = 5000.0  # millisecond


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Generate trace files.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to save the model.")
    # parser.add_argument('--trace-file', type=str, required=True,
    #                     help='Path to trace file')
    parser.add_argument('--trace-dir', type=str, required=True,
                        help='Path to trace file')
    args, unknown = parser.parse_known_args()
    return args



def convert(timestamps, bandwidths):
    """
    timestamps: s
    bandwidths: Mbps
    """
    ms_series = []
    assert len(timestamps) == len(bandwidths)
    ms_t = 0
    for ts, next_ts, bw in zip(timestamps[0:-1], timestamps[1:], bandwidths[0:-1]):
        pkt_per_ms = bw * 1e6 / 8 / BYTES_PER_PACKET / MILLISEC_IN_SEC

        ms_cnt = 0
        pkt_cnt = 0
        while True:
            ms_cnt += 1
            ms_t += 1
            to_send = np.floor((ms_cnt * pkt_per_ms) - pkt_cnt)
            for i in range(int(to_send)):
                ms_series.append(ms_t)
                # mf.write(str(ms_t) + '\n')

            pkt_cnt += to_send

            if ms_cnt >= (next_ts - ts) * MILLISEC_IN_SEC:
                break
    return ms_series

    # files = os.listdir(IN_FILE)
    # for trace_file in files:
    #     with open(IN_FILE + trace_file, 'r') as f, open(OUT_FILE + trace_file, 'w') as mf:
    #         millisec_time = 0
    #         mf.write(str(millisec_time) + '\n')
    #         for line in f:
    #             throughput = float(line.split()[0])
    #             pkt_per_millisec = throughput / BYTES_PER_PKT / MILLISEC_IN_SEC
    #
    #             millisec_count = 0
    #             pkt_count = 0
    #             while True:
    #                 millisec_count += 1
    #                 millisec_time += 1
    #                 to_send = (millisec_count * pkt_per_millisec) - pkt_count
    #                 to_send = np.floor(to_send)
    #
    #                 for i in range(int(to_send)):
    #                     mf.write(str(millisec_time) + '\n')
    #
    #                 pkt_count += to_send
    #
    #                 if millisec_count >= EXP_LEN:
    #                     break
def main():
    args = parse_args()
    for trace_file in glob.glob(os.path.join(args.trace_dir, "*.json")):
        trace_name = os.path.splitext(os.path.basename(trace_file))[0]
        tr =Trace.load_from_file(trace_file)
        ms_series = convert(tr.timestamps, tr.bandwidths)
        with open(os.path.join(args.save_dir, trace_name), 'w', 1) as f:
            for ms in ms_series:
                f.write(str(ms) + '\n')

        with open(os.path.join(args.save_dir, 'loss'), 'w', 1) as f:
            f.write(str(tr.loss_rate))
        with open(os.path.join(args.save_dir, 'queue'), 'w', 1) as f:
            f.write(str(int(tr.queue_size)))
        with open(os.path.join(args.save_dir, 'delay'), 'w', 1) as f:
            f.write(str(int(np.mean(np.array(tr.delays)))))
    # convert()


if __name__ == '__main__':
    main()
