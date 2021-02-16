import argparse
import itertools
import os
import sys
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from stable_baselines import PPO1
from stable_baselines.common.policies import FeedForwardPolicy, MlpPolicy

from common.utils import read_json_file
from simulator import network_sim
from udt_plugins.testing.loaded_agent import LoadedModel

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Pcc Aurora Testing in simulator.")
    parser.add_argument('--model-path', type=str, required=True,
                        help="path to tensorflow model")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to testing results.")
    parser.add_argument('--config', type=str, required=True,
                        help="path to config file in json format.")
    # parser.add_argument('--arch', type=str, default="32,16", help='arch.')
    parser.add_argument('--seed', type=int, default=42, help='seed')

    return parser.parse_args()


class MyMlpPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, **_kwargs):
        arch = [32, 16]
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env,
                                          n_steps, n_batch, reuse, net_arch=[
                                              {"pi": arch, "vf": arch}],
                                          feature_extraction="mlp", **_kwargs)


args = parse_args()

env = gym.make('PccNs-v0', log_dir='')

if args.model_path.endswith('.ckpt'):
    # model is a tensorflow checkpoint
    model = PPO1(MyMlpPolicy, env, verbose=1, schedule='constant',
                 timesteps_per_actorbatch=8192, optim_batchsize=2048,
                 gamma=0.99)
    with model.graph.as_default():
        saver = tf.train.Saver()  # save neural net parameters
        nn_model = os.path.join(args.model_path)
        saver.restore(model.sess, nn_model)
else:
    # model is a tensorflow model to serve
    model = LoadedModel(args.model_path)

#
# bw_list = [1, 5, 10, 100, 500, 1000, 2000, 5000, 8000]
# bw_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# bw_list = [50, 80, 100, 300, 500, 800, 1000, 1500, 2000, 3000, 4000, 5000]
# bw_list = [100]
config = read_json_file(args.config)
bw_list = config['bandwidth']
lat_list = config['latency']
queue_list = config['queue']
loss_list = config['loss']
# bw_list = [1000]
# bw_list = [50]
# lat_list = [0.05]
# queue_list = [5]
# queue_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# loss_list = [0]

param_sets = itertools.product(bw_list, lat_list, loss_list, queue_list)
param_set_len = len(list(param_sets))
param_sets = itertools.product(bw_list, lat_list, loss_list, queue_list)

for env_cnt, (bw, lat, loss, queue) in enumerate(param_sets):
    print(bw, lat, loss, queue)
    env.set_ranges(bw, bw, lat, lat, loss, loss, queue, queue)
    obs = env.reset()
    t_start = time.time()

    while True:
        if isinstance(model, LoadedModel):
            action = model.act(obs.reshape(1, -1))
            action = action['act'][0]
        else:
            action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            os.makedirs(os.path.join(args.save_dir, "rl_test"), exist_ok=True)
            env.dump_events_to_file(
                os.path.join(args.save_dir, "rl_test", "rl_test_log{}.json".format(env_cnt)))
            print("{}/{}, {}s".format(env_cnt, param_set_len, time.time() - t_start))
            t_start = time.time()
            break
