import argparse
import csv
import os
import time
import warnings
import logging
import types

import gym
import numpy as np
import tensorflow as tf
if type(tf.contrib) != types.ModuleType:  # if it is LazyLoader
    tf.contrib._warning = None
from common.utils import set_tf_loglevel
# from simulator.network_simulator import network
# from simulator import network
from simulator.aurora import Aurora
from simulator.trace import generate_traces

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
    parser.add_argument("--config-file", type=str,
                        required=True, help='config file.')

    return parser.parse_args()


def main():
    args = parse_args()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    test_traces = generate_traces(args.config_file, 10, args.duration, args.seed)
    training_traces = generate_traces(args.config_file, 10, args.duration, args.seed)
    # env = gym.make('PccNs-v0', log_dir=args.save_dir, duration=args.duration)
    # env.seed(args.seed)

    aurora = Aurora(training_traces, timesteps_per_actorbatch=10,
                    log_dir=args.save_dir, seed=args.seed,
                    pretrained_model_path=args.model_path)
    results = aurora.test(test_traces)

    for trace, result in zip(test_traces, results):
        with open(os.path.join(args.save_dir, "env" +str(trace) + ".csv"), 'w', 1) as f:
            log_writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            log_writer.writerow(['mean_validation_reward', 'loss',
                                 'throughput', 'latency', 'sending_rate'])
            log_writer.writerows(result)


if __name__ == "__main__":
    main()
