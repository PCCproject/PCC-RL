import argparse
import os
import time
import warnings

import gym
import numpy as np
from common.utils import set_seed

from simulator import good_network_sim as network_sim
from simulator.trace import generate_trace, generate_traces

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
    parser.add_argument('--mtu', type=int, default=1500, help="MTU.")
    # parser.add_argument('--arch', type=str, default="32,16", help='arch.')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--duration', type=int, default=None,
                        help='Flow duration in seconds.')
    parser.add_argument("--config-file", type=str, default=None,
                        help='config file.')

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.config_file is not None:
        test_traces = generate_traces(args.config_file, 10, args.duration)
    else:
        test_traces = [generate_trace((args.duration, args.duration),
                                      (args.bandwidth, args.bandwidth),
                                      (args.delay, args.delay),
                                      (args.loss, args.loss),
                                      (args.queue, args.queue))]
    for _, trace in enumerate(test_traces):
        # log_path = os.path.join(args.save_dir,
        #                         "env_{:.3f}_{:.3f}_{:.3f}_{:.3f}.csv".format(
        #                             trace.bandwidths[0], trace.delay,
        #                             trace.loss_rate, trace.queue_size))
        env = gym.make('PccNs-v0', traces=[trace], congestion_control_type='cubic',
                       log_dir=args.save_dir)
        env.seed(args.seed)

        obs = env.reset()
        rewards = []
        while True:
            action = [0, 0]
            obs, reward, dones, info = env.step(action)
            rewards.append(reward)
            if dones:
                if args.save_dir:
                    env.dump_events_to_file(
                        os.path.join(args.save_dir, "cubic_test_log.json"))
                # print("{}/{}, {}s".format(env_cnt,
                #                           param_set_len, time.time() - t_start))
                break
        # print(rewards)
        print(np.mean(np.array(rewards)))


if __name__ == "__main__":
    main()
