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

import csv
import heapq
import json
import os
import random
import sys
import time

import gym
import ipdb
import numpy as np
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding

from common import config, sender_obs
from simulator.network_simulator.constants import (BITS_PER_BYTE,
                                                   BYTES_PER_PACKET)
from simulator.network_simulator.link import Link
from simulator.network_simulator.sender import Sender, TCPCubicSender

MAX_CWND = 5000
MIN_CWND = 4

MAX_RATE = 1000
MIN_RATE = 40

REWARD_SCALE = 0.001

# MAX_STEPS = 1000
MAX_STEPS = 100
# MAX_STEPS = 3000

EVENT_TYPE_SEND = 'S'
EVENT_TYPE_ACK = 'A'


LATENCY_PENALTY = 1.0
LOSS_PENALTY = 1.0

USE_LATENCY_NOISE = False
# USE_LATENCY_NOISE = True
MAX_LATENCY_NOISE = 1.01

# DEBUG = True
DEBUG = False
START_SENDING_RATE = 500  # packets per second


class Network():

    def __init__(self, senders, links):
        self.event_count = 0
        self.q = []
        self.cur_time = 0.0
        self.senders = senders
        self.links = links
        self.queue_initial_packets()
        self.result_log = []

    def queue_initial_packets(self):
        for sender in self.senders:
            sender.register_network(self)
            sender.reset_obs()
            # heapq.heappush(self.q, (round(1.0 / sender.rate, 5), sender,
            #                         EVENT_TYPE_SEND, 0, 0.0, False, self.event_count, sender.rto, 0))
            heapq.heappush(self.q, (0, sender,
                                    EVENT_TYPE_SEND, 0, 0.0, False, self.event_count, sender.rto, 0))
            self.event_count += 1

    def reset(self):
        self.cur_time = 0.0
        self.q = []
        [link.reset() for link in self.links]
        [sender.reset() for sender in self.senders]
        self.queue_initial_packets()
        self.result_log = []

    def get_cur_time(self):
        return self.cur_time

    # @profile
    def run_for_dur(self):
        dur = self.senders[0].compute_mi_duration()
        end_time = self.cur_time + dur
        # print(self.cur_time, dur)
        mi_id = self.senders[0].create_mi(self.cur_time, end_time)
        # for sender in self.senders:
        #     sender.reset_obs()
        while self.cur_time < end_time:
            (event_time, sender, event_type, next_hop, cur_latency, dropped,
             event_id, rto, pkt_mi_id) = heapq.heappop(self.q)
            if event_time >= end_time:
                heapq.heappush(self.q, (event_time, sender, event_type, next_hop, cur_latency, dropped,
                                        event_id, rto, pkt_mi_id))
                break
            if DEBUG:
                # print("Got %d event %s, to link %d, latency %f at time %f,
                # next_hop %d, dropped %s, event_q length %f, sender rate %f,
                # duration: %f" % ( event_id, event_type, next_hop,
                # cur_latency, event_time, next_hop, dropped, len(self.q),
                # sender.rate, dur))
                print("%d,%s,%d,%f,%f,%d,%s,%f,%f,%f" % (
                      event_id, event_type, next_hop, cur_latency, event_time,
                      next_hop, dropped, len(self.q), sender.rate, -1))
            self.cur_time = event_time
            new_event_time = event_time
            new_event_type = event_type
            new_next_hop = next_hop
            new_latency = cur_latency
            new_dropped = dropped
            push_new_event = False
            # if rto >= 0 and cur_latency > rto:#  sender.timeout(cur_latency):
            #     sender.timeout()
            # new_dropped = True
            # TODO: call TCP timeout logic
            if event_type == EVENT_TYPE_ACK:
                if next_hop == len(sender.path):
                    # if cur_latency > 1.0:
                    #     sender.timeout(cur_latency)
                    # import ipdb
                    # ipdb.set_trace()
                    # sender.on_packet_lost(cur_latency)
                    if dropped:
                        sender.on_packet_lost(pkt_mi_id)
                        # print("Packet lost at time {}, cwnd={}".format(self.cur_time, self.senders[0].cwnd))
                    else:
                        sender.on_packet_acked(cur_latency, pkt_mi_id)
                        # print("Packet acked at time {}, cwnd={}".format(self.cur_time, self.senders[0].cwnd))
                else:
                    new_next_hop = next_hop + 1
                    link_latency = sender.path[next_hop].get_cur_latency(
                        self.cur_time)
                    if USE_LATENCY_NOISE:
                        link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                    new_latency += round(link_latency, 5)
                    new_event_time += round(link_latency, 5)
                    push_new_event = True
            if event_type == EVENT_TYPE_SEND:
                if next_hop == 0:
                    # print("Packet sent at time %f" % self.cur_time)
                    if sender.can_send_packet():
                        sender.on_packet_sent(pkt_mi_id)
                        push_new_event = True
                    if self.cur_time + round(1.0 / sender.rate, 5) < end_time:
                        heapq.heappush(self.q, (self.cur_time + round(1.0 / sender.rate, 5), sender,
                                                EVENT_TYPE_SEND, 0, 0.0, False, self.event_count, sender.rto, mi_id))
                        self.event_count += 1
                    else:
                        heapq.heappush(self.q, (self.cur_time + round(1.0 / sender.rate, 5), sender,
                                                EVENT_TYPE_SEND, 0, 0.0, False, self.event_count, sender.rto, mi_id+1))
                        self.event_count += 1

                else:
                    push_new_event = True

                if next_hop == sender.dest:
                    new_event_type = EVENT_TYPE_ACK
                new_next_hop = next_hop + 1

                link_latency = sender.path[next_hop].get_cur_latency(
                    self.cur_time)
                if USE_LATENCY_NOISE:
                    link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                new_latency += round(link_latency, 5)
                new_event_time += round(link_latency, 5)
                new_dropped = not sender.path[next_hop].packet_enters_link(
                    self.cur_time)

            if push_new_event:
                if new_event_type == EVENT_TYPE_SEND:
                    heapq.heappush(self.q,
                                   (round(new_event_time, 5), sender, new_event_type,
                                    new_next_hop, new_latency, new_dropped,
                                    self.event_count, rto, mi_id))
                    rto = sender.rto
                    self.event_count += 1
                elif new_event_type == EVENT_TYPE_ACK:
                    heapq.heappush(self.q,
                                   (round(new_event_time, 5), sender, new_event_type,
                                    new_next_hop, new_latency, new_dropped,
                                    event_id, rto, pkt_mi_id))

        sender_mi = self.senders[0].get_run_data()
        # print(sender_mi.bytes_sent, sender_mi.bytes_acked, sender_mi.bytes_lost,
        #       sender_mi.send_end - sender_mi.send_start, sender_mi.recv_end - sender_mi.recv_start,
        #             )#, sender_mi.rtt_samples )
        throughput = sender_mi.get("recv rate")  # bits/sec
        send_throughput = sender_mi.get("send rate")  # bits/sec
        latency = sender_mi.get("avg latency")
        loss = sender_mi.get("loss ratio")
        bw_cutoff = self.links[0].bw * 0.8
        lat_cutoff = 2.0 * self.links[0].dl * 1.5
        loss_cutoff = 2.0 * self.links[0].lr * 1.5
        #print("thpt %f, bw %f" % (throughput, bw_cutoff))
        #reward = 0 if (loss > 0.1 or throughput < bw_cutoff or latency > lat_cutoff or loss > loss_cutoff) else 1 #

        # Super high throughput
        #reward = REWARD_SCALE * (20.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)

        # Very high thpt
        reward = (10.0 * throughput / (BITS_PER_BYTE * BYTES_PER_PACKET) -
                  1e3 * latency - 2e3 * loss)
        try:
            ssthresh = self.senders[0].ssthresh
        except:
            ssthresh = 0

        # print("{}, {}, {}, {}, {}, {}, {}".format(
        #     self.cur_time, self.senders[0].cwnd, ssthresh,
        #     self.links[0].bw, self.links[1].bw,
        #     throughput/(8 * BYTES_PER_PACKET), reward, loss))
        self.result_log.append([
            self.cur_time, self.senders[0].cwnd, ssthresh, self.senders[0].rto,
            self.links[0].bw, self.links[1].bw,
            send_throughput / (BITS_PER_BYTE * BYTES_PER_PACKET),
            throughput/(BITS_PER_BYTE * BYTES_PER_PACKET), reward, loss, latency])
        self.cur_time = round(end_time, 5)

        # print(self.cur_time, self.senders[0].cwnd, ssthresh, self.senders[0].rto,
        #     self.links[0].bw, self.links[1].bw,
        #     send_throughput / (BITS_PER_BYTE * BYTES_PER_PACKET),
        #     throughput/(BITS_PER_BYTE * BYTES_PER_PACKET), reward, loss, latency)

        # High thpt
        #reward = REWARD_SCALE * (5.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)

        # Low latency
        #reward = REWARD_SCALE * (2.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        # if reward > 857:
        #print("Reward = %f, thpt = %f, lat = %f, loss = %f" % (reward, throughput, latency, loss))

        #reward = (throughput / RATE_OBS_SCALE) * np.exp(-1 * (LATENCY_PENALTY * latency / LAT_OBS_SCALE + LOSS_PENALTY * loss))
        return reward * REWARD_SCALE


class SimulatedNetworkEnv(gym.Env):

    def __init__(self, history_len=10,
                 features="sent latency inflation,latency ratio,send ratio",
                 congestion_control_type="rl", log_dir=""):
        """Network environment used in simulation.
        congestion_control_type: rl is pcc-rl. cubic is TCPCubic.
        """
        assert congestion_control_type in {"rl", "cubic"}, \
            "Unrecognized congestion_control_type {}.".format(
                congestion_control_type)
        random.seed(42)
        self.log_dir = log_dir
        self.congestion_control_type = congestion_control_type
        if self.congestion_control_type == 'rl':
            self.use_cwnd = False
        elif self.congestion_control_type == 'cubic':
            self.use_cwnd = True
        self.viewer = None
        self.rand = None

        self.min_bw, self.max_bw = (100, 500)  # packet per second
        self.min_lat, self.max_lat = (0.05, 0.5)  # latency second
        self.min_queue, self.max_queue = (0, 8)
        self.min_loss, self.max_loss = (0.0, 0.05)
        self.min_mss, self.max_mss = (1500, 1500)
        self.history_len = history_len
        # print("History length: %d" % history_len)
        self.features = features.split(",")
        # print("Features: %s" % str(self.features))

        self.links = None
        self.senders = None
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links)
        # self.run_dur = None
        self.run_period = 0.1
        self.steps_taken = 0
        self.max_steps = MAX_STEPS
        self.debug_thpt_changes = False
        self.last_thpt = None
        self.last_rate = None

        if self.use_cwnd:
            self.action_space = spaces.Box(
                np.array([-1e12, -1e12]), np.array([1e12, 1e12]), dtype=np.float32)
        else:
            self.action_space = spaces.Box(
                np.array([-1e12]), np.array([1e12]), dtype=np.float32)

        self.observation_space = None
        use_only_scale_free = True
        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                            np.tile(single_obs_max_vec,
                                                    self.history_len),
                                            dtype=np.float32)

        self.reward_sum = 0.0
        self.reward_ewma = 0.0

        self.event_record = {"Events": []}
        self.episodes_run = -1
        print('event_id,event_type,next_hop,cur_latency,event_time,next_hop,dropped,event_q_length,send_rate,duration')

    def set_ranges(self, min_bw, max_bw, min_lat, max_lat, min_loss, max_loss, min_queue, max_queue, min_mss, max_mss):
        self.min_bw = min_bw
        self.max_bw = max_bw
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_loss = min_loss
        self.max_loss = max_loss
        self.min_queue = min_queue
        self.max_queue = max_queue
        self.min_mss = min_mss
        self.max_mss = max_mss

    def seed(self, seed=None):
        self.rand, seed = seeding.np_random(seed)
        return [seed]

    def _get_all_sender_obs(self):
        sender_obs = self.senders[0].get_obs()
        sender_obs = np.array(sender_obs).reshape(-1,)
        # print(sender_obs)
        return sender_obs

    # @profile
    def step(self, actions):
        # print("Actions: %s" % str(actions))
        # print(actions)
        t_start = time.time()
        for i in range(0, 1):  # len(actions)):
            #print("Updating rate for sender %d" % i)
            action = actions
            self.senders[i].apply_rate_delta(action[0])
            if self.use_cwnd:
                self.senders[i].apply_cwnd_delta(action[1])
        valid = False
        for sender in self.senders:
        #     # sender.record_run()
            valid = sender.has_finished_mi()
        #print("Running for %fs" % self.run_dur)
        sender_obs = self._get_all_sender_obs()
        reward = self.net.run_for_dur()
        sender_mi = self.senders[0].get_run_data()
        self.steps_taken += 1
        event = {}
        event["Name"] = "Step"
        event["Time"] = self.steps_taken
        event["Reward"] = reward
        #event["Target Rate"] = sender_mi.target_rate
        event["Send Rate"] = sender_mi.get("send rate")
        event["Throughput"] = sender_mi.get("recv rate")
        event["Latency"] = sender_mi.get("avg latency")
        event["Loss Rate"] = sender_mi.get("loss ratio")
        event["Latency Inflation"] = sender_mi.get("sent latency inflation")
        event["Latency Ratio"] = sender_mi.get("latency ratio")
        event["Send Ratio"] = sender_mi.get("send ratio")
        #event["Cwnd"] = sender_mi.cwnd
        #event["Cwnd Used"] = sender_mi.cwnd_used
        # print(
        #
        # event["Send Rate"] = sender_mi.get("send rate")
        # event["Throughput"] = sender_mi.get("recv rate")
        # event["Latency"] = sender_mi.get("avg latency")
        # event["Loss Rate"] = sender_mi.get("loss ratio")
        # event["Latency Inflation"] = sender_mi.get("sent latency inflation")
        # event["Latency Ratio"] = sender_mi.get("latency ratio")
        # event["Send Ratio"] = sender_mi.get("send ratio")
        #
        #
        #         )
        self.event_record["Events"].append(event)
        # if event["Latency"] > 0.0:
        #     self.run_dur = max(0.5 * sender_mi.get("avg latency"),
        #                        5 / self.senders[0].rate)
        #print("Sender obs: %s" % sender_obs)

        should_stop = False

        self.reward_sum += reward
        # print('env step: {:.4f}s, result_log: {}, mi_cache: {}, {}, network event queue: {} sender rtt samples: {}'.format(
        #     time.time() - t_start, len(self.net.result_log),
        #     len(self.senders[0].mi_cache), str(self.links[0]), len(self.net.q), len(self.senders[0].rtt_samples)))
        # h.heap()
        return sender_obs, reward, (self.steps_taken >= self.max_steps or should_stop), {"valid": valid}

    def print_debug(self):
        print("---Link Debug---")
        for link in self.links:
            link.print_debug()
        print("---Sender Debug---")
        for sender in self.senders:
            sender.print_debug()

    def create_new_links_and_senders(self):
        bw = random.uniform(self.min_bw, self.max_bw)
        lat = random.uniform(self.min_lat, self.max_lat)
        queue = 1 + int(np.exp(random.uniform(self.min_queue, self.max_queue)))
        loss = random.uniform(self.min_loss, self.max_loss)
        # bw    = 1100
        # lat   = 0.05
        # queue = 5
        # queue = 1 + int(np.exp(2))
        # loss  = 0.01
        self.links = [Link(bw, lat, queue, loss), Link(bw, lat, queue, loss)]
        #self.senders = [Sender(0.3 * bw, [self.links[0], self.links[1]], 0, self.history_len)]
        #self.senders = [Sender(random.uniform(0.2, 0.7) * bw, [self.links[0], self.links[1]], 0, self.history_len)]
        if self.congestion_control_type == "rl":
            self.senders = [Sender(START_SENDING_RATE,
                                   # TODO: check dest
                                   [self.links[0], self.links[1]], 0,
                                   self.features,
                                   history_len=self.history_len)]
            # self.senders = [Sender(random.uniform(0.3, 1.5) * bw,
            #                        [self.links[0], self.links[1]], 0,
            #                        self.features,
            #                        history_len=self.history_len)]
            # else:
        elif self.congestion_control_type == "cubic":
            self.senders = [TCPCubicSender(10,
                                          [self.links[0], self.links[1]], 0,
                                          self.features,
                                          history_len=self.history_len)]
        else:
            raise RuntimeError("Unrecognized congestion_control_type {}".format(
                self.congestion_control_type))
        # self.run_dur = 3 * lat
        # self.run_dur = max(0.5 * 0, 5 / self.senders[0].rate)

    def reset(self):
        self.steps_taken = 0
        self.net.reset()
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links)
        self.episodes_run += 1
        if self.episodes_run > 0 and self.episodes_run % 100 == 0:
            self.dump_events_to_file(
                os.path.join(self.log_dir, "pcc_env_log_run_%d.json" % self.episodes_run))
        self.event_record = {"Events": []}
        self.net.run_for_dur()
        # _, _, _, _, valid =
        # self.net.run_for_dur()
        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        # print("Reward: %0.2f, Ewma Reward: %0.2f" % (self.reward_sum, self.reward_ewma))
        self.reward_sum = 0.0
        return self._get_all_sender_obs()#, False

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def dump_events_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.event_record, f, indent=4)
        with open(os.path.splitext(filename)[0]+'.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['timestamp', 'cwnd', 'ssthresh', "rto",
                             'link0_bw', 'link1_bw', "send_throughput",
                             'throughput', 'reward', 'loss', 'latency'])

            writer.writerows(self.net.result_log)


register(id='PccNs-v0', entry_point='simulator.network_simulator.network:SimulatedNetworkEnv')
