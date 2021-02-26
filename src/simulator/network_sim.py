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
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding
import ipdb
import numpy as np

from common import config, sender_obs

MAX_CWND = 5000
MIN_CWND = 4

MAX_RATE = 1000
MIN_RATE = 40

REWARD_SCALE = 0.001

# MAX_STEPS = 400
# MAX_STEPS = 600
MAX_STEPS = 3000

EVENT_TYPE_SEND = 'S'
EVENT_TYPE_ACK = 'A'

BYTES_PER_PACKET = 1500

LATENCY_PENALTY = 1.0
LOSS_PENALTY = 1.0

USE_LATENCY_NOISE = False
MAX_LATENCY_NOISE = 1.1

DEBUG = True
# DEBUG = False

start_int_id = 0


class Link():

    def __init__(self, bandwidth, delay, queue_size, loss_rate):
        self.bw = float(bandwidth)
        self.dl = delay
        self.lr = loss_rate
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
        self.max_queue_delay = queue_size / self.bw
        self.queue_size = queue_size

    def get_cur_queue_delay(self, event_time):
        cur_queue_delay = max(0.0, self.queue_delay - (event_time - self.queue_delay_update_time))
        # print('Event Time: {}s, queue_delay: {}s Queue_delay_update_time: {}s, cur_queue_delay: {}s, max_queue_delay: {}s, bw: {}, queue size: {}'.format(
        #     event_time, self.queue_delay, self.queue_delay_update_time, cur_queue_delay, self.max_queue_delay, self.bw, self.queue_size))
        return cur_queue_delay

    def get_cur_latency(self, event_time):
        q_delay = self.get_cur_queue_delay(event_time)
        # print('queue delay: ', q_delay)
        return self.dl + q_delay

    def packet_enters_link(self, event_time):
        if (random.random() < self.lr):
            return False
        self.queue_delay = self.get_cur_queue_delay(event_time)
        self.queue_delay_update_time = event_time
        extra_delay = 1.0 / self.bw
        # print("Extra delay:{}, Current delay: {}, Max delay: {}".format(extra_delay, self.queue_delay, self.max_queue_delay))
        if extra_delay + self.queue_delay > self.max_queue_delay:
            # print("\tDrop!")
            return False
        self.queue_delay += extra_delay
        # print("\tNew delay = {}".format(self.queue_delay))
        return True

    def print_debug(self):
        print("Link:")
        print("Bandwidth: %f" % self.bw)
        print("Delay: %f" % self.dl)
        print("Queue Delay: %f" % self.queue_delay)
        print("Max Queue Delay: %f" % self.max_queue_delay)
        print("One Packet Queue Delay: %f" % (1.0 / self.bw))

    def reset(self):
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0

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
            heapq.heappush(self.q, (1.0 / sender.rate, sender, EVENT_TYPE_SEND, 0, 0.0, False, self.event_count, sender.rto))
            self.event_count += 1

    def reset(self):
        self.cur_time = 0.0
        self.q = []
        [link.reset() for link in self.links]
        [sender.reset() for sender in self.senders]
        self.queue_initial_packets()

    def get_cur_time(self):
        return self.cur_time

    def run_for_dur(self, dur):
        end_time = self.cur_time + dur
        for sender in self.senders:
            sender.reset_obs()
        while self.cur_time < end_time:
            event_time, sender, event_type, next_hop, cur_latency, dropped, event_id, rto = heapq.heappop(self.q)
            if DEBUG:
                # print("Got %d event %s, to link %d, latency %f at time %f, next_hop %d, dropped %s, event_q length %f, sender rate %f, duration: %f" % (
                #       event_id, event_type, next_hop, cur_latency, event_time, next_hop, dropped, len(self.q), sender.rate, dur))
                print("%d,%s,%d,%f,%f,%d,%s,%f,%f,%f" % (
                      event_id, event_type, next_hop, cur_latency, event_time, next_hop, dropped, len(self.q), sender.rate, dur))
            self.cur_time = event_time
            new_event_time = event_time
            new_event_type = event_type
            new_next_hop = next_hop
            new_latency = cur_latency
            new_dropped = dropped
            push_new_event = False
            # if rto >= 0 and cur_latency > rto:#  sender.timeout(cur_latency):
            #     sender.timeout()
            #     new_dropped = True
            # TODO: call TCP timeout logic
            if event_type == EVENT_TYPE_ACK:
                if next_hop == len(sender.path):
                    # if cur_latency > 1.0:
                    #     sender.timeout(cur_latency)
                        # import ipdb
                        # ipdb.set_trace()
                        # sender.on_packet_lost(cur_latency)
                    if dropped:
                        sender.on_packet_lost(cur_latency)
                        # print("Packet lost at time {}, cwnd={}".format(self.cur_time, self.senders[0].cwnd))
                    else:
                        sender.on_packet_acked(cur_latency)
                        # print("Packet acked at time {}, cwnd={}".format(self.cur_time, self.senders[0].cwnd))
                else:
                    new_next_hop = next_hop + 1
                    link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                    if USE_LATENCY_NOISE:
                        link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                    new_latency += link_latency
                    new_event_time += link_latency
                    push_new_event = True
            if event_type == EVENT_TYPE_SEND:
                if next_hop == 0:
                    # print("Packet sent at time %f" % self.cur_time)
                    if sender.can_send_packet():
                        sender.on_packet_sent()
                        push_new_event = True
                    heapq.heappush(self.q, (self.cur_time + (1.0 / sender.rate), sender, EVENT_TYPE_SEND, 0, 0.0, False, self.event_count, sender.rto))
                    self.event_count += 1

                else:
                    push_new_event = True

                if next_hop == sender.dest:
                    new_event_type = EVENT_TYPE_ACK
                new_next_hop = next_hop + 1

                link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                if USE_LATENCY_NOISE:
                    link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                new_latency += link_latency
                new_event_time += link_latency
                new_dropped = not sender.path[next_hop].packet_enters_link(self.cur_time)

            if push_new_event:
                if new_event_type == EVENT_TYPE_SEND:
                    event_id_push = self.event_count
                    rto = sender.rto
                    self.event_count += 1
                elif new_event_type == EVENT_TYPE_ACK:
                    event_id_push = event_id
                heapq.heappush(self.q, (new_event_time, sender, new_event_type, new_next_hop, new_latency, new_dropped, event_id_push, rto))

        sender_mi = self.senders[0].get_run_data()
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
        reward = (10.0 * throughput / (8 * BYTES_PER_PACKET) - 1e3 * latency - 2e3 * loss)
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
            send_throughput / (8 * BYTES_PER_PACKET),
            throughput/(8 * BYTES_PER_PACKET), reward, loss, latency])
        # print(self.cur_time)

        # High thpt
        #reward = REWARD_SCALE * (5.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)

        # Low latency
        #reward = REWARD_SCALE * (2.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        #if reward > 857:
        #print("Reward = %f, thpt = %f, lat = %f, loss = %f" % (reward, throughput, latency, loss))

        #reward = (throughput / RATE_OBS_SCALE) * np.exp(-1 * (LATENCY_PENALTY * latency / LAT_OBS_SCALE + LOSS_PENALTY * loss))
        return reward * REWARD_SCALE

class Sender():

    def __init__(self, rate, path, dest, features, cwnd=25, history_len=10):
        self.id = Sender._get_next_id()
        self.starting_rate = rate
        self.rate = rate
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_samples = []
        self.prev_rtt_samples = self.rtt_samples
        self.sample_time = []
        self.net = None
        self.path = path
        self.dest = dest
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.cwnd = cwnd
        self.use_cwnd = False
        self.rto = -1
        self.first_pkt_ack_time = None
        self.last_pkt_ack_time = None
        self.first_pkt_send_time = None
        self.last_pkt_send_time = None

        self.got_data = False  # flag used to track whether data is received in a monitor interval


    _next_id = 1
    def _get_next_id():
        result = Sender._next_id
        Sender._next_id += 1
        return result

    def apply_rate_delta(self, delta):
        # print(config.DELTA_SCALE)
        delta *= config.DELTA_SCALE
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))
        # print("current rate {} after applying delta {}".format(self.rate, delta))
        # print("rate %f" % delta)

    def apply_cwnd_delta(self, delta):
        delta *= config.DELTA_SCALE
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_cwnd(self.cwnd * (1.0 + delta))
        else:
            self.set_cwnd(self.cwnd / (1.0 - delta))

    def can_send_packet(self):
        if self.use_cwnd:
            return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd
        else:
            return True

    def register_network(self, net):
        self.net = net

    def on_packet_sent(self):
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET
        if self.first_pkt_send_time is None:
            self.first_pkt_send_time = self.net.get_cur_time()
        self.last_pkt_send_time = self.net.get_cur_time()

    def on_packet_acked(self, rtt):
        self.got_data = True
        self.acked += 1
        self.rtt_samples.append(rtt)
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET
        if self.first_pkt_ack_time is None:
            self.first_pkt_ack_time = self.net.get_cur_time()
        self.last_pkt_ack_time = self.net.get_cur_time()

    def on_packet_lost(self, rtt):
        self.lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET

    def set_rate(self, new_rate):
        self.rate = new_rate
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        # if self.rate > MAX_RATE:
        #     self.rate = MAX_RATE
        # if self.rate < MIN_RATE:
        #     self.rate = MIN_RATE

    def set_cwnd(self, new_cwnd):
        self.cwnd = int(new_cwnd)
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        # if self.cwnd > MAX_CWND:
        #     self.cwnd = MAX_CWND
        # if self.cwnd < MIN_CWND:
        #     self.cwnd = MIN_CWND

    def record_run(self):
        smi = self.get_run_data()
        smi.debug_print()
        self.history.step(smi)
        # print(self.history.as_array(), file=sys.stderr)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        obs_end_time = self.net.get_cur_time()

        #obs_dur = obs_end_time - self.obs_start_time
        # print("Got %d acks in %f seconds" % (self.acked, obs_dur))
        # print("Sent %d packets in %f seconds" % (self.sent, obs_dur))
        # print("self.rate = %f" % self.rate)
        # print(self.acked, self.sent)
        # print(self.rtt_samples)
        if self.got_data:
            return sender_obs.SenderMonitorInterval(
                self.id,
                bytes_sent=self.sent * BYTES_PER_PACKET,
                bytes_acked=self.acked * BYTES_PER_PACKET,
                bytes_lost=self.lost * BYTES_PER_PACKET,
                # send_start=self.first_pkt_send_time,
                # send_end=self.last_pkt_send_time,
                # recv_start=self.first_pkt_ack_time,
                # recv_end=self.last_pkt_ack_time,
                send_start=self.obs_start_time,
                send_end=obs_end_time,
                recv_start=self.obs_start_time,
                recv_end=obs_end_time,
                rtt_samples=self.rtt_samples,
                packet_size=BYTES_PER_PACKET
            )
        else:
            global start_int_id
            start_int_id += 1
            print([self.net.links[0].dl * 2 - 0.0001 * start_int_id])
            return sender_obs.SenderMonitorInterval(
                self.id,
                bytes_sent=self.sent * BYTES_PER_PACKET,
                bytes_acked=self.sent * BYTES_PER_PACKET,
                bytes_lost=0 * BYTES_PER_PACKET,
                send_start=self.obs_start_time,
                send_end=obs_end_time,
                recv_start=self.obs_start_time,
                recv_end=obs_end_time,
                rtt_samples=[self.net.links[0].dl * 2- 0.0003 * i for i in range(5)],
                packet_size=BYTES_PER_PACKET
            )

    def reset_obs(self):
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.prev_rtt_samples = self.rtt_samples
        self.rtt_samples = []
        self.obs_start_time = self.net.get_cur_time()
        self.first_pkt_ack_time = None
        self.last_pkt_ack_time = None
        self.first_pkt_send_time = None
        self.last_pkt_send_time = None

    def print_debug(self):
        print("Sender:")
        print("Obs: %s" % str(self.get_obs()))
        print("Rate: %f" % self.rate)
        print("Sent: %d" % self.sent)
        print("Acked: %d" % self.acked)
        print("Lost: %d" % self.lost)
        print("Min Latency: %s" % str(self.min_latency))

    def reset(self):
        #print("Resetting sender!")
        self.rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.got_data = False

    def timeout(self):
        # placeholder
        pass


class TCPCubicSender(Sender):
    """Mimic TCPCubic sender behavior.

    Args:
        rate
        path
        dest
        features
        cwnd: congestion window size. Unit: number of packets.
        history_len:
    """
    # used byb Cubic
    tcp_friendliness = 1
    fast_convergence = 1
    beta = 0.2
    C = 0.4

    # used by srtt
    ALPHA = 0.8
    BETA = 1.5

    def __init__(self, rate, path, dest, features, cwnd=10, history_len=10):
        super().__init__(rate, path, dest, features, cwnd, history_len)
        # TCP inital cwnd value is configured to 10 MSS. Refer to
        # https://developers.google.com/speed/pagespeed/service/tcp_initcwnd_paper.pdf

        # slow start threshold, arbitrarily high at start
        self.ssthresh = MAX_CWND
        self.pkt_loss_wait_time = 0
        self.use_cwnd = True
        self.rto = 3  # retransmission timeout (seconds)
        # initialize rto to 3s for waiting SYC packets of TCP handshake
        self.srtt = None # (self.ALPHA * self.srtt) + (1 - self.ALPHA) * rtt)

        self.cubic_reset()

    def cubic_reset(self):
        self.W_last_max = 0
        self.epoch_start = 0
        self.origin_point = 0
        self.dMin = 0
        self.W_tcp = 0  # used in tcp friendliness
        self.K = 0
        # self.acked = 0 # TODO is this one used in Cubic?
        self.ack_cnt = 0

        # In standard, TCP cwnd_cnt is an additional state variable that tracks
        # the number of segments acked since the last cwnd increment; cwnd is
        # incremented only when cwnd_cnt > cwnd; then, cwnd_cnt is set to 0.
        # initialie to 0
        self.cwnd_cnt = 0
        self.pkt_loss_wait_time = 0

    def apply_rate_delta(self, delta):
        # place holder
        #  do nothing
        pass

    def apply_cwnd_delta(self, delta):
        # place holder
        #  do nothing
        pass
        # raise NotImplementedError
        # delta *= config.DELTA_SCALE
        # #print("Applying delta %f" % delta)
        # if delta >= 0.0:
        #     self.set_cwnd(self.cwnd * (1.0 + delta))
        # else:
        #     self.set_cwnd(self.cwnd / (1.0 - delta))

    def on_packet_sent(self):
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET

    def on_packet_acked(self, rtt):
        self.acked += 1
        self.rtt_samples.append(rtt)
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET

        # Added by Zhengxu Xia
        if self.dMin:
            self.dMin = min(self.dMin, rtt)
        else:
            self.dMin = rtt
        if self.cwnd <= self.ssthresh:  # in slow start region
            # print("slow start, inc cwnd by 1")
            self.cwnd += 1
        else:  # in congestion avoidance or max bw probing region
            cnt = self.cubic_update()
            if self.cwnd_cnt > cnt:
                # print("in congestion avoidance, inc cwnd by 1")
                self.cwnd += 1
                self.cwnd_cnt = 0
            else:
                # print("in congestion avoidance, inc cwnd_cnt by 1")
                self.cwnd_cnt += 1
        if not self.rtt_samples and not self.prev_rtt_samples:
            print(self.srtt)
            raise RuntimeError("prev_rtt_samples is empty. TCP session is not constructed successfully!")
        elif not self.rtt_samples:
            avg_sampled_rtt = float(np.mean(np.array(self.prev_rtt_samples)))
        else:
            avg_sampled_rtt = float(np.mean(np.array(self.rtt_samples)))
        self.rate = self.cwnd / avg_sampled_rtt
        if self.pkt_loss_wait_time > 0:
            self.pkt_loss_wait_time -= 1

        # TODO: update RTO
        if self.srtt is None:
            self.srtt = rtt
        else:
            self.srtt = (self.ALPHA * self.srtt) + (1 - self.ALPHA) * rtt
        self.rto = max(1, min(self.BETA * self.srtt, 60))

    def on_packet_lost(self, rtt):
        self.lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET

        if self.pkt_loss_wait_time <= 0:
            self.epoch_start = 0
            if self.cwnd < self.W_last_max and self.fast_convergence:
                self.W_last_max = self.cwnd * (2 - self.beta) / 2
            else:
                self.W_last_max = self.cwnd
            old_cwnd = self.cwnd
            self.cwnd = self.cwnd * (1 - self.beta)
            self.ssthresh = self.cwnd
            # print("packet lost: cwnd change from", old_cwnd, "to", self.cwnd)
            if not self.rtt_samples and not self.prev_rtt_samples:
                raise RuntimeError("prev_rtt_samples is empty. TCP session is not constructed successfully!")
            elif not self.rtt_samples:
                avg_sampled_rtt = float(np.mean(np.array(self.prev_rtt_samples)))
            else:
                avg_sampled_rtt = float(np.mean(np.array(self.rtt_samples)))
            self.rate = self.cwnd / avg_sampled_rtt
            self.pkt_loss_wait_time = int(self.cwnd)

    def cubic_update(self):
        self.ack_cnt += 1
        # assume network current time is tcp_time_stamp
        assert self.net is not None
        tcp_time_stamp = self.net.get_cur_time()
        if self.epoch_start <= 0:
            self.epoch_start = tcp_time_stamp # TODO: check the unit of time
            if self.cwnd < self.W_last_max:
                self.K = np.cbrt((self.W_last_max - self.cwnd)/self.C)
                self.origin_point = self.W_last_max
            else:
                self.K = 0
                self.origin_point = self.cwnd
            self.ack_cnt = 1
            self.W_tcp = self.cwnd
        t = tcp_time_stamp + self.dMin - self.epoch_start
        target = self.origin_point + self.C * (t - self.K)**3
        if target > self.cwnd:
            cnt = self.cwnd / (target - self.cwnd)
        else:
            cnt = 100 * self.cwnd
        # TODO: call friendliness
        return cnt

    def reset(self):
        super().reset()
        self.cubic_reset()
        self.rto = 3  # retransmission timeout (seconds)
        # initialize rto to 3s for waiting SYC packets of TCP handshake
        self.srtt = None # (self.ALPHA * self.srtt) + (1 - self.ALPHA) * rtt)

    def timeout(self):
        # if self.pkt_loss_wait_time <= 0:
        # Refer to https://tools.ietf.org/html/rfc8312#section-4.7
        # self.ssthresh = max(int(self.bytes_in_flight / BYTES_PER_PACKET / 2), 2)
        self.sshthresh = self.cwnd * (1 - self.beta)
        self.cwnd = 1
        if not self.rtt_samples and not self.prev_rtt_samples:
            print(self.srtt)
            # raise RuntimeError("prev_rtt_samples is empty. TCP session is not constructed successfully!")
            avg_sampled_rtt = self.srtt
        elif not self.rtt_samples:
            avg_sampled_rtt = float(np.mean(np.array(self.prev_rtt_samples)))
        else:
            avg_sampled_rtt = float(np.mean(np.array(self.rtt_samples)))
        self.rate = self.cwnd / avg_sampled_rtt
        self.cubic_reset()
        # self.pkt_loss_wait_time = int(self.cwnd)
        # print('timeout rate', self.rate, self.cwnd)
        # return True

    def cubic_tcp_friendliness(self):
        raise NotImplementedError


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
        self.min_lat, self.max_lat = (0.05, 0.5) # latency second
        self.min_queue, self.max_queue = (0, 8)
        self.min_loss, self.max_loss = (0.0, 0.05)
        self.history_len = history_len
        # print("History length: %d" % history_len)
        self.features = features.split(",")
        # print("Features: %s" % str(self.features))

        self.links = None
        self.senders = None
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links)
        self.run_dur = None
        self.run_period = 0.1
        self.steps_taken = 0
        self.max_steps = MAX_STEPS
        self.debug_thpt_changes = False
        self.last_thpt = None
        self.last_rate = None

        if self.use_cwnd:
            self.action_space = spaces.Box(np.array([-1e12, -1e12]), np.array([1e12, 1e12]), dtype=np.float32)
        else:
            self.action_space = spaces.Box(np.array([-1e12]), np.array([1e12]), dtype=np.float32)


        self.observation_space = None
        use_only_scale_free = True
        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                            np.tile(single_obs_max_vec, self.history_len),
                                            dtype=np.float32)

        self.reward_sum = 0.0
        self.reward_ewma = 0.0

        self.event_record = {"Events":[]}
        self.episodes_run = -1
        print('event_id,event_type,next_hop,cur_latency,event_time,next_hop,dropped,event_q_length,send_rate,duration')

    def set_ranges(self, min_bw, max_bw, min_lat, max_lat, min_loss, max_loss, min_queue, max_queue):
        self.min_bw = min_bw
        self.max_bw =  max_bw
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_loss= min_loss
        self.max_loss =  max_loss
        self.min_queue= min_queue
        self.max_queue = max_queue


    def seed(self, seed=None):
        self.rand, seed = seeding.np_random(seed)
        return [seed]

    def _get_all_sender_obs(self):
        sender_obs = self.senders[0].get_obs()
        sender_obs = np.array(sender_obs).reshape(-1,)
        #print(sender_obs)
        return sender_obs

    def step(self, actions):
        # print("Actions: %s" % str(actions))
        # print(actions)
        t_start = time.time()
        for i in range(0, 1):#len(actions)):
            #print("Updating rate for sender %d" % i)
            action = actions
            self.senders[i].apply_rate_delta(action[0])
            if self.use_cwnd:
                self.senders[i].apply_cwnd_delta(action[1])
        print("{}\t\t{}".format(self.run_dur, self.senders[0].rate*8*BYTES_PER_PACKET/1e6), file=sys.stderr)
        reward = self.net.run_for_dur(self.run_dur)
        sender_obs = self._get_all_sender_obs()
        sender_mi = self.senders[0].get_run_data()
        for sender in self.senders:
            sender.record_run()
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
        if self.senders[0].got_data and event["Latency"] > 0.0:
            self.run_dur = max(0.5 * sender_mi.get("avg latency"),
                 5 / self.senders[0].rate);
        else:
            self.run_dur = 5 / self.senders[0].rate
        #print("Sender obs: %s" % sender_obs)

        should_stop = False

        self.reward_sum += reward
        # print('env step: {}s'.format(time.time() - t_start))
        return sender_obs, reward, (self.steps_taken >= self.max_steps or should_stop), {}

    def print_debug(self):
        print("---Link Debug---")
        for link in self.links:
            link.print_debug()
        print("---Sender Debug---")
        for sender in self.senders:
            sender.print_debug()

    def create_new_links_and_senders(self):
        bw    = random.uniform(self.min_bw, self.max_bw)
        lat   = random.uniform(self.min_lat, self.max_lat)
        queue = 1 + int(np.exp(random.uniform(self.min_queue, self.max_queue)))
        loss  = random.uniform(self.min_loss, self.max_loss)
        # bw    = 1100
        # lat   = 0.02
        # queue = 5
        # loss  = 0.0
        self.links = [Link(bw, lat, queue, loss), Link(bw, lat, queue, loss)]
        #self.senders = [Sender(0.3 * bw, [self.links[0], self.links[1]], 0, self.history_len)]
        #self.senders = [Sender(random.uniform(0.2, 0.7) * bw, [self.links[0], self.links[1]], 0, self.history_len)]
        if self.congestion_control_type == "rl":
            self.senders = [Sender(500,
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
        self.run_dur = 3 * lat
        # self.run_dur = max(0.5 * 0, 5 / self.senders[0].rate);

    def reset(self):
        self.steps_taken = 0
        self.net.reset()
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links)
        self.episodes_run += 1
        if self.episodes_run > 0 and self.episodes_run % 100 == 0:
            self.dump_events_to_file(
                os.path.join(self.log_dir, "pcc_env_log_run_%d.json" % self.episodes_run))
        self.event_record = {"Events":[]}
        self.net.run_for_dur(self.run_dur)
        for sender in self.senders:
            sender.record_run()
        # self.net.run_for_dur(self.run_dur) # TODO: remove one
        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        # print("Reward: %0.2f, Ewma Reward: %0.2f" % (self.reward_sum, self.reward_ewma))
        self.reward_sum = 0.0
        return self._get_all_sender_obs()

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


register(id='PccNs-v0', entry_point='simulator.network_sim:SimulatedNetworkEnv')
