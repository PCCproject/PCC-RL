# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import heapq
import time
import random
import json
import socket
from simple_arg_parse import arg_or_default

RESET_INTERVAL = 400

# Rates should be in mbps
MAX_RATE = 1000.0
MIN_RATE = 0.25
STARTING_RATE = 2.0

DELTA_SCALE = 0.025# * 1e-12
#DELTA_SCALE = 0.1

RATE_OBS_SCALE = 0.001
LAT_OBS_SCALE = 0.1

# The monitor interval class used to pass data from the PCC subsystem to
# the machine learning module.
#
class SenderMonitorInterval():
    def __init__(self, rate=0.0, recv_rate=0.0, latency=0.0, loss=0.0, lat_infl=0.0, lat_ratio=1.0, send_ratio=1.0):
        self.rate = rate
        self.recv_rate = recv_rate
        self.latency = latency
        self.loss = loss
        self.lat_infl = lat_infl
        self.lat_ratio = lat_ratio
        self.send_ratio = send_ratio

    # Convert the observation parts of the monitor interval into a numpy array
    def as_array(self, scale_free_only):
        if scale_free_only:
            return np.array([self.lat_infl, self.lat_ratio, self.send_ratio])
        else:
            return np.array([self.rate, self.recv_rate, self.latency, self.loss, self.lat_infl,
                self.lat_ratio, self.send_ratio])

class SenderHistory():
    def __init__(self, length, scale_free_only=True):
        self.scale_free_only = scale_free_only
        self.values = []
        self.length = length
        for i in range(0, length):
            self.values.append(SenderMonitorInterval())

    def step(self, new_mi):
        self.values.pop(0)
        self.values.append(new_mi)

    def reset(self):
        self.values = []
        for i in range(0, self.length):
            self.values.append(SenderMonitorInterval())

    def as_array(self):
        arrays = []
        for mi in self.values:
            arrays.append(mi.as_array(self.scale_free_only))
        arrays = np.array(arrays).flatten()
        return arrays

class ShimNetworkEnv(gym.Env):
    
    def __init__(self, history_len=arg_or_default("--history-len", default=10)):
        self.viewer = None
        self.rand = None

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(1)
        self.sock.bind(("localhost", 9787))

        self.conn = None
        self.conn_addr = None
        self.history_len = history_len
        self.history = SenderHistory(history_len)
        self.rate = STARTING_RATE
        self.min_latency = None

        self.action_space = spaces.Box(np.array([-1e12]), np.array([1e12]), dtype=np.float32)
       
        self.observation_space = None
        use_only_scale_free = True 
        if use_only_scale_free:
            self.observation_space = spaces.Box(np.tile(np.array([-1.0, 1.0, 0.0]), self.history_len),
                np.tile(np.array([10.0, 100.0, 1000.0]), self.history_len), dtype=np.float32) 
        else:
            self.observation_space = spaces.Box(np.array([10.0, 0.0, 0.0, 0.0, -100.0]),
                np.array([2000.0, 2000.0, 10.0, 1.0, 10.0]), dtype=np.float32) 

        self.steps_taken = 0
        self.reward_sum = 0.0
        self.reward_ewma = 0.0

    def apply_action(self, action):
        delta = action * DELTA_SCALE
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))

    def set_rate(self, new_rate):
        self.rate = new_rate
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.rate > MAX_RATE:
            self.rate = MAX_RATE
        if self.rate < MIN_RATE:
            self.rate = MIN_RATE

    def seed(self, seed=None):
        self.rand, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.conn is None:
            print("Listening for connection from network sender")
            self.sock.listen()
            self.conn, self.addr = self.sock.accept()
        self.apply_action(action[0])
        self.conn.send(str(self.rate).encode())
        #print("Sent rate, waiting for reward")
        data = self.conn.recv(1024).decode()
        data = data.split("\n")[-2]
        (send_rate, recv_rate, latency, loss, lat_infl, rew) = [float(s) for s in data.split(",")]
        lat_ratio = 0.0
        if latency > 0.0:
            if (self.min_latency is None) or (latency < self.min_latency):
                self.min_latency = latency
            lat_ratio = latency / self.min_latency
        send_ratio = 1000.0
        if (send_rate < 1000.0 * recv_rate) and (recv_rate > 0.0):
            send_ratio = send_rate / recv_rate
        self.history.step(SenderMonitorInterval(
            send_rate * RATE_OBS_SCALE,
            recv_rate * RATE_OBS_SCALE,
            latency * LAT_OBS_SCALE,
            loss,
            lat_infl,
            lat_ratio,
            send_ratio)
        )
        self.reward_sum += rew
        self.steps_taken += 1
        #if self.steps_taken % 10 == 0:
        #    print("Steps taken: %d, Rew: %f" % (self.steps_taken, rew))
        done = (self.steps_taken > RESET_INTERVAL)
        return self.history.as_array(), rew, done, {}

    def reset(self):
        self.min_latency = None
        self.history.reset()
        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        print("Reward: %0.2f, Ewma Reward: %0.2f" % (self.reward_sum, self.reward_ewma))
        self.reward_sum = 0.0
        self.steps_taken = 0
        return self.history.as_array()

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

register(id='NetShim-v0', entry_point='shim_env:ShimNetworkEnv')
#env = SimulatedNetworkEnv()
#env.step([1.0])
