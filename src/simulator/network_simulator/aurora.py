from typing import List, Tuple

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from common import sender_obs
from common.utils import pcc_aurora_reward
from simulator.network_simulator.constants import (
    BITS_PER_BYTE, BYTES_PER_PACKET, MAX_RATE, MI_RTT_PROPORTION, MIN_RATE,
    REWARD_SCALE)
from simulator.network_simulator.link import Link
from simulator.network_simulator.network import Network
from simulator.network_simulator.sender import Sender
from simulator.network_simulator import packet
from simulator.trace import Trace, generate_traces


class AuroraSender(Sender):

    def __init__(self, pacing_rate: float, features: List[str],
                 history_len: int, sender_id: int, dest: int, trace: Trace):
        super().__init__(sender_id, dest)
        self.starting_rate = pacing_rate
        self.pacing_rate = pacing_rate
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.sender_id)
        self.trace = trace
        raise NotImplementedError

    def on_packet_sent(self, pkt: "packet.Packet") -> None:
        return super().on_packet_sent(pkt)

    def on_packet_acked(self, pkt: "packet.Packet") -> None:
        return super().on_packet_acked(pkt)

    def on_packet_lost(self, pkt: "packet.Packet") -> None:
        return super().on_packet_lost(pkt)

    def apply_rate_delta(self, delta):
        if delta >= 0.0:
            self.set_rate(self.pacing_rate * (1.0 + delta))
        else:
            self.set_rate(self.pacing_rate / (1.0 - delta))

    def set_rate(self, new_rate):
        self.pacing_rate = new_rate
        if self.pacing_rate > MAX_RATE:
            self.pacing_rate = MAX_RATE
        if self.pacing_rate < MIN_RATE:
            self.pacing_rate = MIN_RATE

    def record_run(self):
        smi = self.get_run_data()
        self.history.step(smi)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        obs_end_time = self.get_cur_time()

        if not self.rtt_samples and self.prev_rtt_samples:
            rtt_samples = [np.mean(self.prev_rtt_samples)]
        else:
            rtt_samples = self.rtt_samples
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
            self.sender_id,
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

    def on_mi_start(self):
        self.reset_obs()

    def on_mi_finish(self) -> Tuple[float, float]:
        self.record_run()

        sender_mi = self.history.back()  # get_run_data()
        throughput = sender_mi.get("recv rate")  # bits/sec
        latency = sender_mi.get("avg latency")  # second
        loss = sender_mi.get("loss ratio")
        reward = pcc_aurora_reward(
            throughput / BITS_PER_BYTE / BYTES_PER_PACKET, latency, loss,
            self.trace.avg_bw * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET,
            self.trace.avg_delay * 2 / 1e3)

        if latency > 0.0:
            self.mi_duration = MI_RTT_PROPORTION * \
                sender_mi.get("avg latency") + np.mean(extra_delays)
        return reward * REWARD_SCALE, self.mi_duration

    def reset_obs(self):
        self.sent = 0
        self.acked = 0
        self.lost = 0
        if self.rtt_samples:
            self.prev_rtt_samples = self.rtt_samples
        self.rtt_samples = []
        self.rtt_samples_ts = []
        self.queue_delay_samples = []
        self.obs_start_time = self.get_cur_time()

    def reset(self):
        self.pacing_rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.sender_id)
        self.estRTT = 1000000 / 1e6  # SynInterval in emulation
        self.RTTVar = self.estRTT / 2  # RTT variance

        self.got_data = False


class SimulatedNetworkEnv(gym.Env):

    def __init__(self, traces: List[Trace], history_len: int = 10,
                 # features="sent latency inflation,latency ratio,send ratio",
                 features: List[str] = ["sent latency inflation",
                                        "latency ratio", "recv ratio"],
                 train_flag: bool = False, config_file=None,
                 record_pkt_log: bool = False):
        """Network environment used in simulation."""
        self.record_pkt_log = record_pkt_log
        self.config_file = config_file
        self.traces = traces
        self.current_trace = np.random.choice(self.traces)
        self.train_flag = train_flag

        self.history_len = history_len
        self.features = features

        # construct sender and network
        self.links = [Link(self.current_trace), Link(self.current_trace)]
        self.senders = [AuroraSender(
            10 / (self.current_trace.get_delay(0) * 2/1000), self.features,
            self.history_len, 0, 0, self.current_trace)]
        self.net = Network(self.senders, self.links, self.record_pkt_log)
        self.run_dur = 0.01
        self.steps_taken = 0

        self.action_space = spaces.Box(
            np.array([-1e12]), np.array([1e12]), dtype=np.float32)

        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(
            np.tile(single_obs_min_vec, self.history_len),
            np.tile(single_obs_max_vec, self.history_len), dtype=np.float32)

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

    def step(self, action):
        assert self.senders
        self.senders[0].apply_rate_delta(action)
        self.senders[0].on_mi_start()
        self.net.run(self.run_dur)
        # TODO: one run stops and aurora sender should do something
        reward, self.run_dur = self.senders[0].on_mi_finish()
        self.steps_taken += 1
        sender_obs = self._get_all_sender_obs()

        should_stop = self.current_trace.is_finished(self.net.get_cur_time())

        self.reward_sum += reward
        return sender_obs, reward, should_stop, {}

    def reset(self):
        self.steps_taken = 0
        self.net.reset()
        self.current_trace = np.random.choice(self.traces)
        self.current_trace.reset()
        self.senders[0].trace = self.current_trace
        if not self.senders[0].rtt_samples:
            self.run_dur = 0.01
        self.net = Network(self.senders, self.links, self.record_pkt_log)
        self.episodes_run += 1
        if self.train_flag and self.config_file is not None and self.episodes_run % 100 == 0:
            self.traces = generate_traces(self.config_file, 10, duration=30)
        # # TODO: figure out what should happen after reset
        self.senders[0].on_mi_start()
        self.net.run(self.run_dur)
        _, self.run_dur = self.senders[0].on_mi_finish()

        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        self.reward_sum = 0.0
        return self._get_all_sender_obs()


# register(id='PccNs-v0', entry_point='simulator.network:SimulatedNetworkEnv')
