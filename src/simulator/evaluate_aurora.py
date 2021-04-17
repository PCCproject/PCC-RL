import argparse
import csv
import logging
import os
import types
import warnings

import tensorflow as tf

if type(tf.contrib) != types.ModuleType:  # if it is LazyLoader
    tf.contrib._warning = None
from common.utils import set_tf_loglevel, set_seed

from simulator.aurora import Aurora
from simulator.trace import generate_trace, generate_traces

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore")

set_tf_loglevel(logging.FATAL)


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Aurora Testing in simulator.")
    parser.add_argument('--save-dir', type=str, default="",
                        help="direcotry to testing results.")
    parser.add_argument('--model-path', type=str, required=True,
                        help="path to Aurora model.")
    # parser.add_argument('--arch', type=str, default="32,16", help='arch.')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--duration', type=int, default=None,
                        help='Flow duration in seconds.')
    parser.add_argument("--config-file", type=str, default=None,
                        help='config file.')

    parser.add_argument('--delay', type=float, default=50,
                        help="one-way delay. Unit: millisecond.")
    parser.add_argument('--bandwidth', type=float, default=2,
                        help="Constant bandwidth. Unit: mbps.")
    parser.add_argument('--loss', type=float, default=0,
                        help="Constant random loss of uplink.")
    parser.add_argument('--queue', type=int, default=10,
                        help="Uplink queue size. Unit: packets.")
    parser.add_argument('--delta-scale', type=float, default=0.05,
                        help="Environment delta scale.")
    parser.add_argument('--time-variant-bw', action='store_true',
                        help='Generate time variant bandwidth if specified.')

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.config_file is not None:
        test_traces = generate_traces(args.config_file, 1, args.duration,
                                      constant_bw=not args.time_variant_bw)
    else:
        test_traces = [generate_trace((args.duration, args.duration),
                                      (args.bandwidth, args.bandwidth),
                                      (args.delay, args.delay),
                                      (args.loss, args.loss),
                                      (args.queue, args.queue),
                                      constant_bw=not args.time_variant_bw)]

    aurora = Aurora(args.seed, timesteps_per_actorbatch=10,
                    log_dir=args.save_dir,
                    pretrained_model_path=args.model_path,
                    delta_scale=args.delta_scale)
    results, pkt_logs = aurora.test(test_traces)

    for pkt_log in pkt_logs:
        with open(os.path.join(args.save_dir, "aurora_packet_log.csv"), 'w', 1) as f:
            pkt_logger = csv.writer(f, lineterminator='\n')
            pkt_logger.writerows(pkt_log)


if __name__ == "__main__":
    main()
