import argparse
import os
import time
import warnings

import gym
import numpy as np
from common.utils import set_seed

from simulator import good_network_sim as network_sim
from simulator.trace import generate_trace, generate_traces, Trace

warnings.filterwarnings("ignore")


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Cubic Testing in simulator.")
    parser.add_argument('--save-dir', type=str, default="",
                        help="direcotry to testing results.")
    parser.add_argument('--delay', type=float, default=50,
                        help="one-way delay. Unit: millisecond.")
    parser.add_argument('--bandwidth', type=float, default=2,
                        help="Constant bandwidth. Unit: mbps.")
    parser.add_argument('--loss', type=float, default=0,
                        help="Constant random loss of uplink.")
    parser.add_argument('--queue', type=int, default=100,
                        help="Uplink queue size. Unit: packets.")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--duration', type=int, default=30,
                        help='Flow duration in seconds.')
    parser.add_argument("--config-file", type=str, default=None,
                        help='config file.')
    parser.add_argument('--time-variant-bw', action='store_true',
                        help='Generate time variant bandwidth if specified.')
    parser.add_argument('--trace-file', type=str, default=None,
                        help='path to a trace file.')

    return parser.parse_args()


def test_on_trace(trace, save_dir, seed):
    env = gym.make('PccNs-v0', traces=[trace], congestion_control_type='cubic',
                   log_dir=save_dir)
    env.seed(seed)

    _ = env.reset()
    rewards = []
    while True:
        action = [0, 0]
        _, reward, dones, _ = env.step(action)
        rewards.append(reward)
        if dones:
            # if args.save_dir:
            #     env.dump_events_to_file(
            #         os.path.join(args.save_dir, "cubic_test_log.json"))
            # print("{}/{}, {}s".format(env_cnt,
            #                           param_set_len, time.time() - t_start))
            break
    return rewards


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.trace_file:
        test_traces = [Trace.load_from_file(args.trace_file)]
    elif args.config_file is not None:
        test_traces = generate_traces(args.config_file, 1, args.duration,
                                      constant_bw=not args.time_variant_bw)
    else:
        test_traces = [generate_trace((args.duration, args.duration),
                                      (args.bandwidth, args.bandwidth),
                                      (args.delay, args.delay),
                                      (args.loss, args.loss),
                                      (args.queue, args.queue),
                                      constant_bw=not args.time_variant_bw)]
    for _, trace in enumerate(test_traces):
        rewards = test_on_trace(trace, args.save_dir, args.seed)
        print(np.mean(np.array(rewards)))


if __name__ == "__main__":
    main()
