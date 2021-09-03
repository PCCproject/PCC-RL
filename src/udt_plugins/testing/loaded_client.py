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

import argparse
import csv
import os
import random
import sys
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
print(sys.path, file=sys.stderr, flush=True)

import numpy as np

from common import sender_obs
# from udt_plugins.testing import loaded_agent
from simulator.aurora import Aurora


MIN_RATE = 0.06
MAX_RATE = 300.0
DELTA_SCALE = 1  # 0.05

RESET_RATE_MIN = 5.0
RESET_RATE_MAX = 100.0

RESET_RATE_MIN = 1.2
RESET_RATE_MAX = 1.2
def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Loaded_client")
    parser.add_argument('--model-path', type=str, required=True,
                        help="path to a NN model.")
    parser.add_argument('--save-dir', type=str, default="",
                        help="path to save logs.")
    parser.add_argument("--history-len", type=int, default=10,
                        help="Feature history length.")
    parser.add_argument("--input-features", type=str,
                        # default=["sent latency inflation", "latency ratio",
                        #          "send ratio"],
                        default=["sent latency inflation", "latency ratio",
                                 "recv ratio"], nargs=3, help="Feature type.")

    args, unknown = parser.parse_known_args()
    return args


class PccGymDriver():

    flow_lookup = {}

    def __init__(self, flow_id):
        global RESET_RATE_MIN
        global RESET_RATE_MAX
        args = parse_args()

        self.id = flow_id

        self.rate = random.uniform(RESET_RATE_MIN, RESET_RATE_MAX)
        self.history_len = args.history_len
        self.features = args.input_features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features,
                                                self.id)
        self.save_dir = args.save_dir
        self.log_writer = csv.writer(open(os.path.join(
            self.save_dir, 'aurora_emulation_log.csv'), 'w', 1),
            lineterminator='\n')
        self.log_writer.writerow(['timestamp', "target_send_rate", "send_rate",
                                  'recv_rate', 'latency', 'loss', 'reward',
                                  "action", "bytes_sent", "bytes_acked",
                                  'bytes_lost', 'send_start_time',
                                  "send_end_time", 'recv_start_time',
                                  'recv_end_time', 'latency_increase',
                                  "sent_latency_inflation", 'latency_ratio',
                                  'send_ratio', 'recv_ratio', 'packet_size',
                                  "min_rtt", 'rtt_samples'])
        self.got_data = False

        # self.agent = loaded_agent.LoadedModelAgent(args.model_path)
        self.aurora = Aurora(seed=20, log_dir="", timesteps_per_actorbatch=10,
                             pretrained_model_path=args.model_path,
                             delta_scale=DELTA_SCALE)

        PccGymDriver.flow_lookup[flow_id] = self

        self.t_start = time.time()
        # dummpy inference here to load model
        # _ = self.agent.act(self.history.as_array())
        _ = self.aurora.model.predict(self.history.as_array(), deterministic=True)

        self.mi_pushed = False

        self.idx = 1

    def get_rate(self):
        if self.has_data() and self.mi_pushed:
            # rate_delta = self.agent.act(self.history.as_array())
            # rate_delta = self.agent.act(self.sim_features[self.idx])
            t_start = time.time()
            rate_delta, _ = self.aurora.model.predict(self.history.as_array(), deterministic=True)

            rate_delta = rate_delta.item()
            target_rate = self.rate
            # self.rate = apply_rate_delta(self.rate, rate_delta)
            # print('get rate costs {:.4f}'.format(time.time() - t_start), old_rate, self.rate, rate_delta, self.history.as_array(), file=sys.stderr, flush=True)
            try:
                mi = self.history.values[-1]
                # sim_rate_delta = self.actions[self.idx]
                # self.rate = apply_rate_delta(self.rate, rate_delta)
                # print(rate_delta, sim_rate_delta, file=sys.stderr, flush=True)
                self.idx += 1
                send_rate = mi.get("send rate")
                recv_rate = mi.get("recv rate")
                sent_lat_inf = mi.get("sent latency inflation")
                latency_ratio = mi.get("latency ratio")
                send_ratio = mi.get("send ratio")
                recv_ratio = mi.get("recv ratio")
                latency = mi.get("avg latency")
                loss_rate = mi.get("loss ratio")
                latency_increase = mi.get("latency increase")
                min_lat = mi.get("conn min latency")
                reward = 10.0 * recv_rate / \
                    (8 * mi.packet_size) - 1e3 * latency - 2e3 * loss_rate
                self.log_writer.writerow([
                    mi.send_end, target_rate * 1e6, send_rate, recv_rate,
                    latency, loss_rate, reward, rate_delta, mi.bytes_sent,
                    mi.bytes_acked, mi.bytes_lost, mi.send_start, mi.send_end,
                    mi.recv_start, mi.recv_end, latency_increase, sent_lat_inf,
                    latency_ratio, send_ratio, recv_ratio, mi.packet_size,
                    min_lat, mi.rtt_samples])
                self.rate = apply_rate_delta(send_rate / 1e6, rate_delta)
            except Exception as e:
                print(e, file=sys.stderr, flush=True)
        else:
            rate_delta = 0
        self.mi_pushed = False
        return self.rate * 1e6

    def has_data(self):
        return self.got_data

    def set_current_rate(self, new_rate):
        self.current_rate = new_rate

    def record_observation(self, new_mi):
        self.history.step(new_mi)
        self.got_data = True

    def reset_rate(self):
        self.current_rate = random.uniform(RESET_RATE_MIN, RESET_RATE_MAX)

    def reset_history(self):
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features,
                                                self.id)
        self.got_data = False

    def reset(self):
        # self.agent.reset()
        self.reset_rate()
        self.reset_history()

    def give_sample(self, bytes_sent, bytes_acked, bytes_lost,
                    send_start_time, send_end_time, recv_start_time,
                    recv_end_time, rtt_samples, packet_size, utility):
        self.record_observation(
            sender_obs.SenderMonitorInterval(
                self.id,
                bytes_sent=bytes_sent,
                bytes_acked=bytes_acked,
                bytes_lost=bytes_lost,
                send_start=send_start_time,
                send_end=send_end_time,
                recv_start=recv_start_time,
                recv_end=recv_end_time,
                rtt_samples=rtt_samples,
                packet_size=packet_size, #sender_obs.MAXIMUM_SEGMENT_SIZE,
            )
        )
        self.mi_pushed = True

    def get_by_flow_id(flow_id):
        return PccGymDriver.flow_lookup[flow_id]


def give_sample(flow_id, bytes_sent, bytes_acked, bytes_lost,
                send_start_time, send_end_time, recv_start_time,
                recv_end_time, rtt_samples, packet_size, utility):
    driver = PccGymDriver.get_by_flow_id(flow_id)
    driver.give_sample(bytes_sent, bytes_acked, bytes_lost,
                       send_start_time, send_end_time, recv_start_time,
                       recv_end_time, rtt_samples, packet_size, utility)


def apply_rate_delta(rate, rate_delta):
    global MIN_RATE
    global MAX_RATE
    global DELTA_SCALE

    rate_delta *= DELTA_SCALE

    # We want a string of actions with average 0 to result in a rate change
    # of 0, so delta of 0.05 means rate * 1.05,
    # delta of -0.05 means rate / 1.05
    if rate_delta > 0:
        rate *= (1.0 + rate_delta)
    elif rate_delta < 0:
        rate /= (1.0 - rate_delta)

    # For practical purposes, we may have maximum and minimum rates allowed.
    if rate < MIN_RATE:
        rate = MIN_RATE
    if rate > MAX_RATE:
        rate = MAX_RATE

    return rate


def reset(flow_id):
    driver = PccGymDriver.get_by_flow_id(flow_id)
    driver.reset()


def get_rate(flow_id):
    driver = PccGymDriver.get_by_flow_id(flow_id)
    return driver.get_rate()


def init(flow_id):
    driver = PccGymDriver(flow_id)
