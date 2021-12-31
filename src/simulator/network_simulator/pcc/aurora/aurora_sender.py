from typing import List, Tuple

import numpy as np

from common import sender_obs
from common.utils import pcc_aurora_reward
from simulator.network_simulator.constants import (
    BITS_PER_BYTE, BYTES_PER_PACKET, MAX_RATE, MI_RTT_PROPORTION, MIN_RATE)
from simulator.network_simulator.sender import Sender
from simulator.network_simulator import packet
from simulator.trace import Trace

class AuroraSender(Sender):

    def __init__(self, pacing_rate: float, features: List[str],
                 history_len: int, sender_id: int, dest: int, trace: Trace):
        super().__init__(sender_id, dest)
        self.starting_rate = pacing_rate
        self.pacing_rate = pacing_rate
        self.pacing_rate = pacing_rate
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.sender_id)
        self.trace = trace
        self.got_data = False
        self.cwnd = 0
        self.prev_rtt_samples = []

    def on_packet_sent(self, pkt: "packet.Packet") -> None:
        return super().on_packet_sent(pkt)

    def on_packet_acked(self, pkt: "packet.Packet") -> None:
        super().on_packet_acked(pkt)
        self.rtt_samples_ts.append(self.get_cur_time())
        if not self.got_data:
            self.got_data = len(self.rtt_samples) >= 1

    def on_packet_lost(self, pkt: "packet.Packet") -> None:
        return super().on_packet_lost(pkt)

    def apply_rate_delta(self, delta):
        if delta >= 0.0:
            self.set_rate(self.pacing_rate * (1.0 + delta))
        else:
            self.set_rate(self.pacing_rate / (1.0 - delta))

    def set_rate(self, new_rate):
        self.pacing_rate = new_rate
        if self.pacing_rate > MAX_RATE * BYTES_PER_PACKET:
            self.pacing_rate = MAX_RATE * BYTES_PER_PACKET
        if self.pacing_rate < MIN_RATE * BYTES_PER_PACKET:
            self.pacing_rate = MIN_RATE * BYTES_PER_PACKET

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

    def schedule_send(self, first_pkt: bool = False, on_ack: bool = False):
        assert self.net, "network is not registered in sender."
        if first_pkt:
            next_send_time = 0
        else:
            next_send_time = self.get_cur_time() + BYTES_PER_PACKET / self.pacing_rate
        next_pkt = packet.Packet(next_send_time, self, 0)
        self.net.add_packet(next_pkt)

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
                sender_mi.get("avg latency") # + np.mean(extra_delays)
        return reward, self.mi_duration

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
        self.prev_rtt_samples = []
