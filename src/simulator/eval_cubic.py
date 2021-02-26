import argparse
import itertools
import os
import time

import gym
import ipdb
import matplotlib.pyplot as plt
import numpy as np

from common.utils import read_json_file
from simulator import good_network_sim as network_sim


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Cubic Testing in simulator.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to testing results.")
    parser.add_argument('--config', type=str, required=True,
                        help="path to config file in json format.")
    # parser.add_argument('--arch', type=str, default="32,16", help='arch.')
    parser.add_argument('--seed', type=int, default=42, help='seed')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    env = gym.make('PccNs-v0', congestion_control_type='cubic',
                   log_dir=args.save_dir)
    env.seed(args.seed)

    # in dist
    # bw_list = [100, 300, 500]
    # lat_list = [0.05, 0.3, 0.5]
    # queue_list = [1, 3, 5, 7]
    # loss_list = [0.0, 0.02, 0.04]

    # out dist
    # bw_list = [500, 700, 900]
    # lat_list = [0.05, 0.3, 0.5]
    # queue_list = [1, 3, 5, 7]
    # loss_list = [0.0, 0.02, 0.04]
    # bw_list = [1, 5, 10, 100, 500, 1000, 2000, 5000, 8000]
    # bw_list = [50, 80, 100, 300, 500, 800, 1000, 1500, 2000, 3000, 4000, 5000]
    # bw_list = [100]  # , 2000, 5000, 8000]
    # lat_list = [0.05]
    # queue_list = [5]
    # queue_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # queue_list = [8]
    # loss_list = [0.0]
    config = read_json_file(args.config)
    bw_list = config['bandwidth']
    lat_list = config['latency']
    queue_list = config['queue']
    loss_list = config['loss']

    param_sets = itertools.product(bw_list, lat_list, loss_list, queue_list)
    param_set_len = len(list(param_sets))
    param_sets = itertools.product(bw_list, lat_list, loss_list, queue_list)
    for env_cnt, (bw, lat, loss, queue) in enumerate(param_sets):
        print(bw, lat, loss, queue)
        env.set_ranges(bw, bw, lat, lat, loss, loss, queue, queue)
        obs = env.reset()
        t_start = time.time()
        while True:
            # action, _states = model.predict(obs)
            # print(action.shape)
            action = [0, 0]
            obs, rewards, dones, info = env.step(action)
            # print(rewards, dones, info, step_cnt, env.run_dur,
            # env.links[0].print_debug(), env.links[1].print_debug(),
            # env.senders)
            if dones:
                env.dump_events_to_file(
                    os.path.join(args.save_dir,
                                 "cubic_test_log{}.json".format(env_cnt)))
                print("{}/{}, {}s".format(env_cnt,
                                          param_set_len, time.time() - t_start))
                t_start = time.time()
                break


if __name__ == "__main__":
    main()
