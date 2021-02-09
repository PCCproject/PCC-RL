import os
import gym
import numpy as np
import itertools
import time

import sys
sys.path.insert(0, "../udt-plugins/testing")

from loaded_agent import LoadedModel

import network_sim
# model = LoadedModel('./icml_paper_model')
model = LoadedModel('/tmp/pcc_saved_models/model_A')
env = gym.make('PccNs-v0')

# OUT_DIR = "../../results/test_rl_in_dist"
OUT_DIR = "../../results/test_rl_out_dist1"

os.makedirs(OUT_DIR, exist_ok=True)

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

bw_list = [1000, 1100, 1200, 1300, 1400, 1500]
lat_list = [0.02]
queue_list = [5]
loss_list = [0.0]

for env_cnt, (bw, lat, loss, queue) in enumerate(
    itertools.product(bw_list, lat_list, loss_list, queue_list)):
    env.set_ranges(bw, bw, lat, lat, loss, loss, queue, queue)
    obs = env.reset()
    t_start = time.time()

    while True:
        action = model.act(obs.reshape(1, -1))
        # print(action['act'][0].shape)
        obs, rewards, dones, info = env.step(action['act'][0])
        # print(rewards, dones, info, step_cnt, env.run_dur, env.links[0].print_debug(), env.links[1].print_debug(), env.senders)
        if dones:
            env.dump_events_to_file(
                os.path.join(OUT_DIR, "rl_test_log{}.json".format(env_cnt)))
            print(env_cnt, time.time() - t_start)
            t_start = time.time()
            break
