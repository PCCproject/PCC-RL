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
import heapq
import os
import random
import sys
import time
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
from typing import Tuple, List

import gym
import numpy as np
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding

from common import sender_obs
from common.utils import pcc_aurora_reward
from simulator.network_simulator.constants import (BITS_PER_BYTE, BYTES_PER_PACKET, EVENT_TYPE_ACK,
                                 EVENT_TYPE_SEND, MAX_RATE, MI_RTT_PROPORTION,
                                 MIN_RATE, REWARD_SCALE)
from simulator.network_simulator.link import Link
from simulator.trace import generate_traces

LATENCY_PENALTY = 1.0
LOSS_PENALTY = 1.0

USE_LATENCY_NOISE = True
MAX_LATENCY_NOISE = 1.1

# DEBUG = True
DEBUG = False


def debug_print(msg):
    if DEBUG:
        print(msg, file=sys.stderr, flush=True)


class Network():

    def __init__(self, senders, links, env):
        self.event_count = 0
        self.q = []
        self.cur_time = 0.0
        self.senders = senders
        self.links = links
        self.queue_initial_packets()
        self.env = env

        self.pkt_log = []

        # self.recv_rate_cache = []

    def queue_initial_packets(self):
        for sender in self.senders:
            sender.register_network(self)
            sender.reset_obs()
            heapq.heappush(self.q, (0, sender, EVENT_TYPE_SEND,
                                    0, 0.0, False, self.event_count, sender.rto, 0))
            self.event_count += 1

    def reset(self):
        self.pkt_log = []
        self.cur_time = 0.0
        self.q = []
        [link.reset() for link in self.links]
        [sender.reset() for sender in self.senders]
        self.queue_initial_packets()
        # self.recv_rate_cache = []

    def get_cur_time(self):
        return self.cur_time

    def run_for_dur(self, dur, action=None):
        if self.senders[0].lat_diff != 0:
            self.senders[0].start_stage = False
        start_time = self.cur_time
        end_time = min(self.cur_time + dur,
                       self.env.current_trace.timestamps[-1])
        # debug_print('MI from {} to {}, dur {}'.format(
        #     self.cur_time, end_time, dur))
        for sender in self.senders:
            sender.reset_obs()
        # set_obs_start = False
        extra_delays = []  # time used to put packet onto the network
        while True:
            event_time, sender, event_type, next_hop, cur_latency, dropped, \
                event_id, rto, event_queue_delay = self.q[0]
            # if not sender.got_data and event_time >= end_time and event_type == EVENT_TYPE_ACK and next_hop == len(sender.path):
            #     end_time = event_time
            #     self.cur_time = end_time
            #     self.env.run_dur = end_time - start_time
            #     break
            if sender.got_data and event_time >= end_time and event_type == EVENT_TYPE_SEND:
                end_time = event_time
                self.cur_time = end_time
                break
            event_time, sender, event_type, next_hop, cur_latency, dropped, \
                event_id, rto, event_queue_delay = heapq.heappop(self.q)
            self.cur_time = event_time
            new_event_time = event_time
            new_event_type = event_type
            new_next_hop = next_hop
            new_latency = cur_latency
            new_dropped = dropped
            new_event_queue_delay = event_queue_delay
            push_new_event = False
            # debug_print("Got %d event %s, to link %d, latency %f at time %f, "
            #             "next_hop %d, dropped %s, event_q length %f, "
            #             "sender rate %f, duration: %f, queue_size: %f, "
            #             "rto: %f, cwnd: %f, ssthresh: %f, sender rto %f, "
            #             "pkt in flight %d, wait time %d" % (
            #                 event_id, event_type, next_hop, cur_latency,
            #                 event_time, next_hop, dropped, len(self.q),
            #                 sender.rate, dur, self.links[0].queue_size,
            #                 rto, sender.cwnd, sender.ssthresh, sender.rto,
            #                 int(sender.bytes_in_flight/BYTES_PER_PACKET),
            #                 sender.pkt_loss_wait_time))
            if event_type == EVENT_TYPE_ACK:
                if next_hop == len(sender.path):
                    # if cur_latency > 1.0:
                    #     sender.timeout(cur_latency)
                    # sender.on_packet_lost(cur_latency)
                    if rto >= 0 and cur_latency > rto and sender.pkt_loss_wait_time <= 0:
                        sender.timeout()
                        dropped = True
                        new_dropped = True
                    elif dropped:
                        sender.on_packet_lost(cur_latency)
                        if self.env.record_pkt_log:
                            self.pkt_log.append(
                                [self.cur_time, event_id, 'lost',
                                 BYTES_PER_PACKET, cur_latency, event_queue_delay,
                                 self.links[0].pkt_in_queue,
                                 sender.rate * BYTES_PER_PACKET * BITS_PER_BYTE,
                                 self.links[0].get_bandwidth(self.cur_time) * BYTES_PER_PACKET * BITS_PER_BYTE])
                    else:
                        sender.on_packet_acked(cur_latency)
                        # debug_print('Ack packet at {}'.format(self.cur_time))
                        # log packet acked
                        if self.env.record_pkt_log:
                            self.pkt_log.append(
                                [self.cur_time, event_id, 'acked',
                                 BYTES_PER_PACKET, cur_latency,
                                 event_queue_delay, self.links[0].pkt_in_queue,
                                 sender.rate * BYTES_PER_PACKET * BITS_PER_BYTE,
                                 self.links[0].get_bandwidth(self.cur_time) * BYTES_PER_PACKET * BITS_PER_BYTE])
                else:
                    # comment out to save disk usage
                    # if self.env.record_pkt_log:
                    #     self.pkt_log.append(
                    #         [self.cur_time, event_id, 'arrived',
                    #          BYTES_PER_PACKET, cur_latency, event_queue_delay,
                    #          self.links[0].pkt_in_queue,
                    #          sender.rate * BYTES_PER_PACKET * BITS_PER_BYTE,
                    #          self.links[0].get_bandwidth(self.cur_time) * BYTES_PER_PACKET * BITS_PER_BYTE])
                    new_next_hop = next_hop + 1
                    # new_event_queue_delay += sender.path[next_hop].get_cur_queue_delay(
                    #     self.cur_time)
                    link_latency = sender.path[next_hop].get_cur_propagation_latency(
                        self.cur_time)
                    # link_latency *= self.env.current_trace.get_delay_noise_replay(self.cur_time)
                    # if USE_LATENCY_NOISE:
                    # link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                    new_latency += link_latency
                    new_event_time += link_latency
                    push_new_event = True
            elif event_type == EVENT_TYPE_SEND:
                if next_hop == 0:
                    if sender.can_send_packet():
                        sender.on_packet_sent()
                        # print('Send packet at {}'.format(self.cur_time))
                        if not self.env.train_flag and self.env.record_pkt_log:
                            self.pkt_log.append(
                                [self.cur_time, event_id, 'sent',
                                 BYTES_PER_PACKET, cur_latency,
                                 event_queue_delay, self.links[0].pkt_in_queue,
                                 sender.rate * BYTES_PER_PACKET * BITS_PER_BYTE,
                                 self.links[0].get_bandwidth(self.cur_time) * BYTES_PER_PACKET * BITS_PER_BYTE])
                        push_new_event = True
                    heapq.heappush(self.q, (self.cur_time + (1.0 / sender.rate),
                                            sender, EVENT_TYPE_SEND, 0, 0.0,
                                            False, self.event_count, sender.rto,
                                            0))
                    self.event_count += 1

                else:
                    push_new_event = True

                if next_hop == sender.dest:
                    new_event_type = EVENT_TYPE_ACK
                new_next_hop = next_hop + 1

                prop_delay, new_event_queue_delay = sender.path[next_hop].get_cur_latency(
                    self.cur_time)
                link_latency = prop_delay + new_event_queue_delay
                # if USE_LATENCY_NOISE:
                # link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                # link_latency += self.env.current_trace.get_delay_noise(
                #     self.cur_time, self.links[0].get_bandwidth(self.cur_time)) / 1000
                # link_latency += max(0, np.random.normal(0, 1) / 1000)
                # link_latency += max(0, np.random.uniform(0, 5) / 1000)
                rand = random.uniform(0, 1)
                if rand > 0.9:
                    noise = random.uniform(0, sender.path[next_hop].trace.delay_noise) / 1000
                else:
                    noise = 0
                new_latency += noise
                new_event_time += noise
                # link_latency *= self.env.current_trace.get_delay_noise_replay(self.cur_time)
                new_latency += link_latency
                new_event_time += link_latency
                new_dropped = not sender.path[next_hop].packet_enters_link(
                    self.cur_time)
                extra_delays.append(
                    1 / self.links[0].get_bandwidth(self.cur_time))
                # new_latency += 1 / self.links[0].get_bandwidth(self.cur_time)
                # new_event_time += 1 / self.links[0].get_bandwidth(self.cur_time)
                if not new_dropped:
                    sender.queue_delay_samples.append(new_event_queue_delay)

            if push_new_event:
                heapq.heappush(self.q, (new_event_time, sender, new_event_type,
                                        new_next_hop, new_latency, new_dropped,
                                        event_id, rto, new_event_queue_delay))
        for sender in self.senders:
            sender.record_run()

        sender_mi = self.senders[0].history.back() #get_run_data()
        throughput = sender_mi.get("recv rate")  # bits/sec
        latency = sender_mi.get("avg latency")  # second
        loss = sender_mi.get("loss ratio")
        # debug_print("thpt %f, delay %f, loss %f, bytes sent %f, bytes acked %f" % (
        #     throughput/1e6, latency, loss, sender_mi.bytes_sent, sender_mi.bytes_acked))
        avg_bw_in_mi = self.env.current_trace.get_avail_bits2send(start_time, end_time) / (end_time - start_time) / BITS_PER_BYTE / BYTES_PER_PACKET
        # avg_bw_in_mi = np.mean(self.env.current_trace.bandwidths) * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET
        reward = pcc_aurora_reward(
            throughput / BITS_PER_BYTE / BYTES_PER_PACKET, latency, loss,
            avg_bw_in_mi, np.mean(self.env.current_trace.delays) * 2 / 1e3)

        # self.env.run_dur = MI_RTT_PROPORTION * self.senders[0].estRTT # + np.mean(extra_delays)
        if latency > 0.0:
            self.env.run_dur = MI_RTT_PROPORTION * \
                sender_mi.get("avg latency") + np.mean(np.array(extra_delays))
        # elif self.env.run_dur != 0.01:
            # assert self.env.run_dur >= 0.03
            # self.env.run_dur = max(MI_RTT_PROPORTION * sender_mi.get("avg latency"), 5 * (1 / self.senders[0].rate))

        # self.senders[0].avg_latency = sender_mi.get("avg latency")  # second
        # self.senders[0].recv_rate = round(sender_mi.get("recv rate"), 3)  # bits/sec
        # self.senders[0].send_rate = round(sender_mi.get("send rate"), 3)  # bits/sec
        # self.senders[0].lat_diff = sender_mi.rtt_samples[-1] - sender_mi.rtt_samples[0]
        # self.senders[0].latest_rtt = sender_mi.rtt_samples[-1]
        # self.recv_rate_cache.append(self.senders[0].recv_rate)
        # if len(self.recv_rate_cache) > 6:
        #     self.recv_rate_cache = self.recv_rate_cache[1:]
        # self.senders[0].max_tput = max(self.recv_rate_cache)
        #
        # if self.senders[0].lat_diff == 0 and self.senders[0].start_stage:  # no latency change
        #     pass
        #     # self.senders[0].max_tput = max(self.senders[0].recv_rate, self.senders[0].max_tput)
        # elif self.senders[0].lat_diff == 0 and not self.senders[0].start_stage:  # no latency change
        #     pass
        #     # self.senders[0].max_tput = max(self.senders[0].recv_rate, self.senders[0].max_tput)
        # elif self.senders[0].lat_diff > 0:  # latency increase
        #     self.senders[0].start_stage = False
        #     # self.senders[0].max_tput = self.senders[0].recv_rate # , self.max_tput)
        # else:  # latency decrease
        #     self.senders[0].start_stage = False
        #     # self.senders[0].max_tput = max(self.senders[0].recv_rate, self.senders[0].max_tput)
        return reward # * REWARD_SCALE


class Sender():

    def __init__(self, rate, path, dest, features, cwnd=25, history_len=10,
                 delta_scale=1):
        self.id = Sender._get_next_id()
        self.delta_scale = delta_scale
        self.starting_rate = rate
        self.rate = rate
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_samples = []
        self.rtt_samples_ts = []
        self.queue_delay_samples = []
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
        self.ssthresh = 0
        self.pkt_loss_wait_time = -1
        # self.estRTT = 1000000 / 1e6  # SynInterval in emulation
        # self.RTTVar = self.estRTT / 2  # RTT variance
        self.estRTT = None  # SynInterval in emulation
        self.RTTVar = None  # RTT variance
        self.got_data = False

        self.min_rtt = 10
        self.max_tput = 0
        self.start_stage = True
        self.lat_diff = 0
        self.recv_rate = 0
        self.send_rate = 0
        self.latest_rtt = 0

        # variables to track accross the connection session
        self.tot_sent = 0 # no. of packets
        self.tot_acked = 0 # no. of packets
        self.tot_lost = 0 # no. of packets
        self.cur_avg_latency = 0.0
        self.first_ack_ts = None
        self.last_ack_ts = None
        self.first_sent_ts = None
        self.last_sent_ts = None

        # variables to track binwise measurements
        self.bin_bytes_sent = {}
        self.bin_bytes_acked = {}
        self.lat_ts = []
        self.lats = []
        self.bin_size = 500 # ms

    _next_id = 1

    @property
    def avg_sending_rate(self):
        """Average sending rate in packets/second."""
        assert self.last_ack_ts is not None and self.first_ack_ts is not None
        assert self.last_sent_ts is not None and self.first_sent_ts is not None
        return self.tot_sent / (self.last_sent_ts - self.first_sent_ts)

    @property
    def avg_throughput(self):
        """Average throughput in packets/second."""
        assert self.last_ack_ts is not None and self.first_ack_ts is not None
        assert self.last_sent_ts is not None and self.first_sent_ts is not None
        return self.tot_acked / (self.last_ack_ts - self.first_ack_ts)

    @property
    def avg_latency(self):
        """Average latency in second."""
        return self.cur_avg_latency

    @property
    def pkt_loss_rate(self):
        """Packet loss rate in one connection session."""
        return 1 - self.tot_acked / self.tot_sent

    @property
    def bin_tput(self) -> Tuple[List[float], List[float]]:
        tput_ts = []
        tput = []
        for bin_id in sorted(self.bin_bytes_acked):
            tput_ts.append(bin_id * self.bin_size / 1000)
            tput.append(
                self.bin_bytes_acked[bin_id] * BITS_PER_BYTE / self.bin_size * 1000 / 1e6)
        return tput_ts, tput

    @property
    def bin_sending_rate(self) -> Tuple[List[float], List[float]]:
        sending_rate_ts = []
        sending_rate = []
        for bin_id in sorted(self.bin_bytes_sent):
            sending_rate_ts.append(bin_id * self.bin_size / 1000)
            sending_rate.append(
                self.bin_bytes_sent[bin_id] * BITS_PER_BYTE / self.bin_size * 1000 / 1e6)
        return sending_rate_ts, sending_rate

    @property
    def latencies(self) -> Tuple[List[float], List[float]]:
        return self.lat_ts, self.lats

    def _get_next_id():
        result = Sender._next_id
        Sender._next_id += 1
        return result

    def apply_rate_delta(self, delta):
        # if self.got_data:
        delta *= self.delta_scale
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))

    def apply_cwnd_delta(self, delta):
        delta *= self.delta_scale
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
        assert self.net
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET
        self.tot_sent += 1
        if self.first_sent_ts is None:
            self.first_sent_ts = self.net.get_cur_time()
        self.last_sent_ts = self.net.get_cur_time()

        bin_id = int((self.net.get_cur_time() - self.first_sent_ts) * 1000 / self.bin_size)
        self.bin_bytes_sent[bin_id] = self.bin_bytes_sent.get(bin_id, 0) + BYTES_PER_PACKET

    def on_packet_acked(self, rtt):
        assert self.net
        self.cur_avg_latency = (self.cur_avg_latency * self.tot_acked + rtt) / (self.tot_acked + 1)
        self.tot_acked += 1
        if self.first_ack_ts is None:
            self.first_ack_ts = self.net.get_cur_time()
        self.last_ack_ts = self.net.get_cur_time()

        self.min_rtt = min(self.min_rtt, rtt)
        if self.estRTT is None and self.RTTVar is None:
            self.estRTT = rtt
            self.RTTVar = rtt / 2
        elif self.estRTT and self.RTTVar:
            self.estRTT = (7.0 * self.estRTT + rtt) / 8.0  # RTT of emulation way
            self.RTTVar = (self.RTTVar * 7.0 + abs(rtt - self.estRTT) * 1.0) / 8.0
        else:
            raise ValueError("srtt and rttvar shouldn't be None.")

        self.acked += 1
        self.rtt_samples.append(rtt)
        self.rtt_samples_ts.append(self.net.get_cur_time())
        # self.rtt_samples.append(self.estRTT)
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET
        if not self.got_data:
            self.got_data = len(self.rtt_samples) >= 1
        # self.got_data = True

        bin_id = int((self.net.get_cur_time() - self.first_ack_ts) * 1000 / self.bin_size)
        self.bin_bytes_acked[bin_id] = self.bin_bytes_acked.get(bin_id, 0) + BYTES_PER_PACKET
        self.lat_ts.append(self.net.get_cur_time())
        self.lats.append(rtt * 1000)

    def on_packet_lost(self, rtt):
        self.lost += 1
        self.tot_lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET

    def set_rate(self, new_rate):
        self.rate = new_rate
        # print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.rate > MAX_RATE:
            self.rate = MAX_RATE
        if self.rate < MIN_RATE:
            self.rate = MIN_RATE

    def set_cwnd(self, new_cwnd):
        self.cwnd = int(new_cwnd)
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        # if self.cwnd > MAX_CWND:
        #     self.cwnd = MAX_CWND
        # if self.cwnd < MIN_CWND:
        #     self.cwnd = MIN_CWND

    def record_run(self):
        smi = self.get_run_data()
        # if not self.got_data and smi.rtt_samples:
        #     self.got_data = True
        #     self.history.step(smi)
        # else:
        self.history.step(smi)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        assert self.net
        obs_end_time = self.net.get_cur_time()

        #obs_dur = obs_end_time - self.obs_start_time
        #print("Got %d acks in %f seconds" % (self.acked, obs_dur))
        #print("Sent %d packets in %f seconds" % (self.sent, obs_dur))
        #print("self.rate = %f" % self.rate)
        # print(self.acked, self.sent)
        if not self.rtt_samples and self.prev_rtt_samples:
            rtt_samples = [np.mean(np.array(self.prev_rtt_samples))]
        else:
            rtt_samples = self.rtt_samples
        # if not self.rtt_samples:
        #     print(self.obs_start_time, obs_end_time, self.rate)
        # rtt_samples is empty when there is no packet acked in MI
        # Solution: inherit from previous rtt_samples.

        # recv_start = self.rtt_samples_ts[0] if len(
        #     self.rtt_samples) >= 2 else self.obs_start_time
        recv_start = self.history.back().recv_end if len(
            self.rtt_samples) >= 1 else self.obs_start_time
        recv_end = self.rtt_samples_ts[-1] if len(
            self.rtt_samples) >= 1 else obs_end_time
        bytes_acked = self.acked * BYTES_PER_PACKET
        if recv_start == 0:
            recv_start = self.rtt_samples_ts[0]
            bytes_acked = (self.acked - 1) * BYTES_PER_PACKET

        # bytes_acked = max(0, (self.acked-1)) * BYTES_PER_PACKET if len(
        #     self.rtt_samples) >= 2 else self.acked * BYTES_PER_PACKET
        return sender_obs.SenderMonitorInterval(
            self.id,
            bytes_sent=self.sent * BYTES_PER_PACKET,
            # max(0, (self.acked-1)) * BYTES_PER_PACKET,
            # bytes_acked=self.acked * BYTES_PER_PACKET,
            bytes_acked=bytes_acked,
            bytes_lost=self.lost * BYTES_PER_PACKET,
            send_start=self.obs_start_time,
            send_end=obs_end_time,
            # recv_start=self.obs_start_time,
            # recv_end=obs_end_time,
            recv_start=recv_start,
            recv_end=recv_end,
            rtt_samples=rtt_samples,
            queue_delay_samples=self.queue_delay_samples,
            packet_size=BYTES_PER_PACKET
        )

    def reset_obs(self):
        assert self.net
        self.sent = 0
        self.acked = 0
        self.lost = 0
        if self.rtt_samples:
            self.prev_rtt_samples = self.rtt_samples
        self.rtt_samples = []
        self.rtt_samples_ts = []
        self.queue_delay_samples = []
        self.obs_start_time = self.net.get_cur_time()

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
        self.estRTT = 1000000 / 1e6  # SynInterval in emulation
        self.RTTVar = self.estRTT / 2  # RTT variance

        self.got_data = False
        self.min_rtt = 10
        self.max_tput = 0
        self.start_stage = True
        self.lat_diff = 0
        self.recv_rate = 0
        self.send_rate = 0
        self.latest_rtt = 0

        self.tot_sent = 0 # no. of packets
        self.tot_acked = 0 # no. of packets
        self.tot_lost = 0 # no. of packets
        self.cur_avg_latency = 0.0
        self.first_ack_ts = None
        self.last_ack_ts = None

        self.bin_bytes_sent = {}
        self.bin_bytes_acked = {}
        self.lat_ts = []
        self.lats = []

    def timeout(self):
        # placeholder
        pass


class SimulatedNetworkEnv(gym.Env):

    def __init__(self, traces, history_len=10,
                 # features="sent latency inflation,latency ratio,send ratio",
                 features="sent latency inflation,latency ratio,recv ratio",
                 train_flag=False, delta_scale=1.0, config_file=None,
                 record_pkt_log: bool = False, real_trace_prob: float = 0):
        """Network environment used in simulation.
        congestion_control_type: aurora is pcc-rl. cubic is TCPCubic.
        """
        self.real_trace_prob = real_trace_prob
        self.record_pkt_log = record_pkt_log
        self.config_file = config_file
        self.delta_scale = delta_scale
        self.traces = traces
        if self.config_file:
            self.current_trace = generate_traces(self.config_file, 1, 30)[0]
        else:
            self.current_trace = np.random.choice(self.traces)
        self.train_flag = train_flag
        self.use_cwnd = False

        self.history_len = history_len
        # print("History length: %d" % history_len)
        self.features = features.split(",")
        # print("Features: %s" % str(self.features))

        self.links = None
        self.senders = None
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links, self)
        self.run_dur = None
        self.run_period = 0.1
        self.steps_taken = 0
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
        # use_only_scale_free = True
        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                            np.tile(single_obs_max_vec,
                                                    self.history_len),
                                            dtype=np.float32)

        self.reward_sum = 0.0
        self.reward_ewma = 0.0

        self.episodes_run = -1

    def seed(self, seed=None):
        self.rand, seed = seeding.np_random(seed)
        return [seed]

    def _get_all_sender_obs(self):
        assert self.senders
        sender_obs = self.senders[0].get_obs()
        sender_obs = np.array(sender_obs).reshape(-1,)
        return sender_obs

    def step(self, actions):
        assert self.senders
        #print("Actions: %s" % str(actions))
        # print(actions)
        for i in range(0, 1):  # len(actions)):
            #print("Updating rate for sender %d" % i)
            action = actions
            self.senders[i].apply_rate_delta(action[0])
            if self.use_cwnd:
                self.senders[i].apply_cwnd_delta(action[1])
        # print("Running for %fs" % self.run_dur)
        reward = self.net.run_for_dur(self.run_dur, action=actions[0])
        self.steps_taken += 1
        sender_obs = self._get_all_sender_obs()

        should_stop = self.current_trace.is_finished(self.net.get_cur_time())

        self.reward_sum += reward
        # print('env step: {}s'.format(time.time() - t_start))

        # sender_obs = np.array([self.senders[0].send_rate,
        #         self.senders[0].avg_latency,
        #         self.senders[0].lat_diff, int(self.senders[0].start_stage),
        #         self.senders[0].max_tput, self.senders[0].min_rtt,
        #         self.senders[0].latest_rtt])
        return sender_obs, reward, should_stop, {}

    def create_new_links_and_senders(self):
        self.links = [Link(self.current_trace), Link(self.current_trace)]
        self.senders = [Sender(
            10 / (self.current_trace.get_delay(0) *2/1000),
                               [self.links[0], self.links[1]], 0,
                               self.features,
                               history_len=self.history_len,
                               delta_scale=self.delta_scale)]
        # self.run_dur = 3 * lat
        # self.run_dur = 1 * lat
        if not self.senders[0].rtt_samples:
            # self.run_dur = 0.473
            # self.run_dur = 5 / self.senders[0].rate
            self.run_dur = 0.01
            # self.run_dur = self.current_trace.get_delay(0) * 2 / 1000

    def reset(self):
        self.steps_taken = 0
        self.net.reset()
        # old snippet start
        # self.current_trace = np.random.choice(self.traces)
        # old snippet end

        # choose real trace with a probability. otherwise, use synthetic trace
        self.current_trace = generate_traces(self.config_file, 1, duration=30)[0]
        if random.uniform(0, 1) < self.real_trace_prob and self.traces:
            real_trace = np.random.choice(self.traces)  # randomly select a real trace
            real_trace.queue_size = self.current_trace.queue_size
            real_trace.loss_rate = self.current_trace.loss_rate
            self.current_trace = real_trace

        # if self.train_flag and not self.config_file:
        #     bdp = np.max(self.current_trace.bandwidths) / BYTES_PER_PACKET / \
        #             BITS_PER_BYTE * 1e6 * np.max(self.current_trace.delays) * 2 / 1000
        #     self.current_trace.queue_size = max(2, int(bdp * np.random.uniform(0.2, 3.0))) # hard code this for now
        #     loss_rate_exponent = float(np.random.uniform(np.log10(0+1e-5), np.log10(0.5+1e-5), 1))
        #     if loss_rate_exponent < -4:
        #         loss_rate = 0
        #     else:
        #         loss_rate = 10**loss_rate_exponent
        #     self.current_trace.loss_rate = loss_rate

        self.current_trace.reset()
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links, self)
        self.episodes_run += 1

        # old code snippet start
        # if self.train_flag and self.config_file is not None and self.episodes_run % 100 == 0:
        #     self.traces = generate_traces(self.config_file, 10, duration=30)
        # old code snippet end
        self.net.run_for_dur(self.run_dur)
        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        self.reward_sum = 0.0
        return self._get_all_sender_obs()


register(id='PccNs-v0', entry_point='simulator.network:SimulatedNetworkEnv')
