import argparse
import itertools
import os
import time

import gym
import ipdb
import matplotlib.pyplot as plt
import numpy as np

from simulator import network_sim

def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Training code.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to save the model.")
    # parser.add_argument('--arch', type=str, default="32,16", help='arch.')
    parser.add_argument('--seed', type=int, default=42, help='seed')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    env = gym.make('PccNs-v0', congestion_control_type='cubic', log_dir=args.save_dir)
    env.seed(args.seed)

    # OUT_DIR = "../../results/test_cubic_in_dist"
    # OUT_DIR = "../../results/test_cubic_out_dist1"
    # OUT_DIR = "./tmp"
    # OUT_DIR = "../../results/sanity_check"
    # OUT_DIR = "../../results/rand_bw/bw_1_10/cubic_test"


    # obs = env.reset()

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
    bw_list = [50, 80, 100, 300, 500, 800, 1000, 1500, 2000, 3000, 4000, 5000]
    # bw_list = [1000] #, 2000, 5000, 8000]
    lat_list = [0.05]
    queue_list = [5]
    loss_list = [0.0]

    param_sets = itertools.product(bw_list, lat_list, loss_list, queue_list)
    param_set_len = len(list(param_sets))
    param_sets = itertools.product(bw_list, lat_list, loss_list, queue_list)
    for env_cnt, (bw, lat, loss, queue) in enumerate(param_sets):
        # if env_cnt == 3:
        #     ipdb.set_trace()
        # env.min_bw, env.max_bw = bw, bw
        # env.min_lat, env.max_lat = lat, lat
        # env.min_loss, env.max_loss = loss, loss
        # env.min_queue, env.max_queue = queue, queue
        print(bw, lat, loss, queue)
        env.set_ranges(bw, bw, lat, lat, loss, loss, queue, queue)
        obs = env.reset()
        t_start = time.time()
        while True:
            # action, _states = model.predict(obs)
            # print(action.shape)
            action = [0, 0]
            obs, rewards, dones, info = env.step(action)
            # print(rewards, dones, info, step_cnt, env.run_dur, env.links[0].print_debug(), env.links[1].print_debug(), env.senders)
            if dones:
                env.dump_events_to_file(
                    os.path.join(args.save_dir, "cubic_test_log{}.json".format(env_cnt)))
                print("{}/{}, {}s".format(env_cnt, param_set_len, time.time() - t_start))
                t_start = time.time()
                break

if __name__ == "__main__":
    main()
