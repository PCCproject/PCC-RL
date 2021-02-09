import itertools
import os
import gym
from stable_baselines import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

import network_sim
import sys

# OUT_DIR = "../../results/rand_bw/bw_1_100"
# OUT_DIR = "../../results/sanity_check"
# OUT_DIR = "../../results/train_10000_steps"
# OUT_DIR = "../../results/train_100000_steps"
# OUT_DIR = "../../results/train_500000_steps"
# OUT_DIR = "../../results/train_1000000_steps"
# OUT_DIR = "../../results/train_2000000_steps"
OUT_DIR = "../../results/rand_bw/bw_50_100"
# OUT_DIR = "../../results/rand_bw/bw_50_500"
# OUT_DIR = "../../results/rand_bw/bw_50_1000"
# OUT_DIR = "../../results/rand_bw/bw_50_1500"
# OUT_DIR = "../../results/rand_bw/bw_50_2000"
# OUT_DIR = "../../results/rand_bw/bw_50_3000"
# OUT_DIR = "../../results/rand_bw/bw_50_5000"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class MyMlpPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, **_kwargs):
        arch = [32, 16]
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env,
                                          n_steps, n_batch, reuse, net_arch=[
                                              {"pi": arch, "vf": arch}],
                                          feature_extraction="mlp", **_kwargs)

env = gym.make('PccNs-v0', log_dir='')

model = PPO1(MyMlpPolicy, env, verbose=1, schedule='constant',
             timesteps_per_actorbatch=8192, optim_batchsize=2048,
             gamma=0.99)
with model.graph.as_default():
    saver = tf.train.Saver()  # save neural net parameters
    nn_model = os.path.join(OUT_DIR, "pcc_model_best.ckpt")
    # nn_model = os.path.join("../../results/tmp/pcc_model_0.ckpt")
    saver.restore(model.sess, nn_model)
# model = PPO1(MyMlpPolicy, env, verbose=1)

# for model_idx in range(6):

# model.learn(total_timesteps=2)
# model.save("ppo1_cartpole")
#
# del model # remove to demonstrate saving and loading
#
# model = PPO1.load("ppo1_cartpole")
#
# bw_list = [1, 5, 10, 100, 500, 1000, 2000, 5000, 8000]
# bw_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
bw_list = [50, 80, 100, 300, 500, 800, 1000, 1500, 2000, 3000, 4000, 5000]
# bw_list = [1000]
# bw_list = [3000]
bw_list = [50]
lat_list = [0.05]
queue_list = [5]
loss_list = [0]

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
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            os.makedirs(os.path.join(OUT_DIR, "rl_test"), exist_ok=True)
            env.dump_events_to_file(
                os.path.join(OUT_DIR, "rl_test", "rl_test_log{}.json".format(env_cnt)))
            print("{}/{}, {}s".format(env_cnt, param_set_len, time.time() - t_start))
            t_start = time.time()
            break
    break
