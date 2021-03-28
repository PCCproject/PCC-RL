import argparse
import os
import warnings
import logging
import types

import tensorflow as tf
if type(tf.contrib) != types.ModuleType:  # if it is LazyLoader
    tf.contrib._warning = None
from common.utils import set_tf_loglevel
# from simulator.network_simulator import network
# from simulator import network
from simulator.aurora import Aurora
from simulator.trace import generate_traces, generate_trace

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
    parser.add_argument('--queue', type=int, default=100,
                        help="Uplink queue size. Unit: packets.")
    parser.add_argument('--delta-scale', type=float, default=0.05,
                        help="Environment delta scale.")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.config_file is not None:
        test_traces = generate_traces(args.config_file, 10, args.duration)
    else:
        test_traces = [generate_trace(args.duration,
                                      (args.bandwidth, args.bandwidth),
                                      (args.delay, args.delay),
                                      (args.loss, args.loss),
                                      (args.queue, args.queue))]

    aurora = Aurora(args.seed, timesteps_per_actorbatch=10,
                    log_dir=args.save_dir,
                    pretrained_model_path=args.model_path,
                    delta_scale=args.delta_scale)
    results = aurora.test(test_traces)

    for _, (trace, result) in enumerate(zip(test_traces, results)):
        log_path = os.path.join(args.save_dir,
                                "env_{:.3f}_{:.3f}_{:.3f}_{:.3f}.csv".format(
                                    trace.bandwidths[0], trace.delay,
                                    trace.loss_rate, trace.queue_size))
        with open(log_path, 'w', 1) as f:
            obs_header = []
            for i in range(10):
                obs_header.append("send_latency_ratio {}".format(i))
                obs_header.append("latency_ratio {}".format(i))
                obs_header.append("send_ratio {}".format(i))

            f.write("\t\t".join(['ts', 'mi', 'reward',
                               'send_rate', 'throughput', 'latency', 'loss',
                               'action'] + obs_header)+ "\n")
            for line in result:
                log_line = "{timestamp:.3f}\t\t{mi:.3f}\t\t{reward:.3f}\t\t{sending_rate:.3f}\t\t" \
                    "{throughput:.3f}\t\t{latency:.3f}\t\t{loss:.3f}\t\t" \
                    "{action:.3f}\t\t".format(
                        timestamp=line[0], mi=line[8], reward=line[1],
                        sending_rate=line[2], throughput=line[3],
                        latency=line[4], loss=line[5], action=line[6]) + '\t\t'.join(["{:.3f}".format(ob) for ob in line[7]]) + "\n"
                f.write(log_line)


if __name__ == "__main__":
    main()
