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
import ast
import heapq
import time
import random
import json
import socket
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir) 
sys.path.insert(0,grandparentdir) 
from common.simple_arg_parse import arg_or_default
from common import sender_obs

RESET_INTERVAL = 400

# Rates should be in mbps
MAX_RATE = 1000.0
MIN_RATE = 0.25
STARTING_RATE = 2.0

DELTA_SCALE = 0.025# * 1e-12
#DELTA_SCALE = 0.1

RATE_OBS_SCALE = 0.001
LAT_OBS_SCALE = 0.1

class ShimNetworkEnv(gym.Env):
    
    def __init__(self, history_len=arg_or_default("--history-len", default=10),
                 features=arg_or_default("--input-features",
                    default="sent latency inflation,"
                          + "latency ratio,"
                          + "send ratio")):
        self.viewer = None
        self.rand = None

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(1)
        self.sock.bind(("localhost", 9787))

        self.conn = None
        self.conn_addr = None
        self.features = features.split(",")
        self.history_len = history_len
        self.history = sender_obs.SenderHistory(history_len, self.features, 0)
        self.rate = STARTING_RATE

        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                            np.tile(single_obs_max_vec, self.history_len),
                                            dtype=np.float32)
        self.action_space = spaces.Box(np.array([-1e12]), np.array([1e12]), dtype=np.float32)

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
        data = self.conn.recv(1024).decode()
        data = data.split("\n")[-2]
        vals = data.split(";")
        flow_id = int(vals[0])
        bytes_sent = int(vals[1])
        bytes_acked = int(vals[2])
        bytes_lost = int(vals[3])
        send_start_time = float(vals[4])
        send_end_time = float(vals[5])
        recv_start_time = float(vals[6])
        recv_end_time = float(vals[7])
        rtt_samples = [float(rtt) for rtt in ast.literal_eval(vals[8])]
        packet_size = int(vals[9])
        rew = float(vals[10])
        
        self.history.step(sender_obs.SenderMonitorInterval(
            flow_id,
            bytes_sent=bytes_sent,
            bytes_acked=bytes_acked,
            bytes_lost=bytes_lost,
            send_start=send_start_time,
            send_end=send_end_time,
            recv_start=recv_start_time,
            recv_end=recv_end_time,
            rtt_samples=rtt_samples,
            packet_size=packet_size
        ))
        self.reward_sum += rew
        self.steps_taken += 1
        done = (self.steps_taken > RESET_INTERVAL)
        return self.history.as_array(), rew, done, {}

    def reset(self):
        self.history = sender_obs.SenderHistory(self.history_len, self.features, 0)
        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        print("Reward: %0.2f, Ewma Reward: %0.2f" % (self.reward_sum, self.reward_ewma))
        self.reward_sum = 0.0
        self.steps_taken = 0
        self.set_rate(STARTING_RATE)
        return self.history.as_array()

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

register(id='NetShim-v0', entry_point='shim_env:ShimNetworkEnv')
