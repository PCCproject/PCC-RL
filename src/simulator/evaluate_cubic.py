import argparse
import os
import time
import warnings

import gym
import numpy as np

from simulator import good_network_sim as network_sim

warnings.filterwarnings("ignore")


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Cubic Testing in simulator.")
    parser.add_argument('--save-dir', type=str, default="",
                        help="direcotry to testing results.")
    # parser.add_argument('--config', type=str, required=True,
    #                     help="path to config file in json format.")
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

    return parser.parse_args()


def main():
    args = parse_args()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    env = gym.make('PccNs-v0', congestion_control_type='cubic',
                   log_dir=args.save_dir, duration=args.duration)
    env.seed(args.seed)

    bw = args.bandwidth / 8 / 1500 * 1e6
    lat = args.delay / 1000
    queue = args.queue
    loss = args.loss

    env.set_ranges(bw, bw, lat, lat, loss, loss, queue, queue)
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
