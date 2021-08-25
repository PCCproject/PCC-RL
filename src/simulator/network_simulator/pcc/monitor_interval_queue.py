from typing import Union
from simulator.network_simulator.packet import Packet
from simulator.network_simulator.pcc.monitor_interval import MonitorInterval
from simulator.network_simulator.pcc.vivace.vivace_latency import VivaceLatencySender


class MonitorIntervalQueue:

    def __init__(self, sender: VivaceLatencySender) -> None:
        self.q = []
        self.sender = sender
        self.num_useful_intervals = 0
        self.num_available_intervals = 0
        self.mi_cnt = 0

    def push(self, mi: MonitorInterval) -> None:
        self.q.append(mi)

    def pop(self) -> MonitorInterval:
        if self.empty():
            raise RuntimeError("MI queue is empty!")
        mi = self.q.pop(0)
        return mi

    def empty(self) -> bool:
        return len(self.q) == 0

    def current(self) -> MonitorInterval:
        if self.empty():
            raise RuntimeError("MI queue is empty!")
        return self.q[-1]

    def front(self) -> MonitorInterval:
        if self.empty():
            raise RuntimeError("MI queue is empty!")
        return self.q[0]

    def on_packet_sent(self, pkt: Packet, sent_interval: float) -> None:
        if self.empty():
            # raise RuntimeError("MI queue is empty!")
            return
        cur_mi = self.q[-1]
        if cur_mi.bytes_sent == 0:
            cur_mi.first_packet_sent_time = pkt.sent_time
            cur_mi.first_packet_id = pkt.pkt_id
        self.q[-1].last_packet_sent_time = pkt.sent_time
        self.q[-1].last_packet_id = pkt.pkt_id
        self.q[-1].bytes_sent += pkt.pkt_size
        self.q[-1].packet_sent_intervals.push_back(sent_interval)

    def on_packet_acked(self, ts: float, pkt_id: int, rtt: float,
                     queue_delay: float) -> None:
        if self.empty():
            raise RuntimeError("MI queue is empty!")
        for mi in self.q:
            if mi.all_pkts_accounted_for(ts):
                continue
            mi.on_pkt_acked(ts, pkt_id, rtt, queue_delay)

    def on_pkt_lost(self, ts, pkt_id) -> None:
        if self.empty():
            raise RuntimeError("MI queue is empty!")
        for mi in self.q:
            if mi.all_pkts_accounted_for(ts):
                continue
            mi.on_pkt_lost(ts, pkt_id)

    def size(self) -> int:
        return len(self.q)

    def has_finished_mi(self, ts) -> bool:
        if self.empty():
            return False
        return self.q[0].all_pkts_accounted_for(ts)

    def get_finished_mi(self, ts) -> Union[MonitorInterval, None]:
        if self.has_finished_mi(ts):
            return self.pop()
        else:
            return None

    def print_mi_ids(self):
        output = ""
        for mi in self.q:
            output += str(mi.mi_id) + ", "
        print(output)

    def extend_current_interval(self):
        if self.empty():
            raise RuntimeError("MI queue is empty!")
        self.q[-1].is_monitor_duration_extended = True

    def enqueue_new_monitor_interval(
        self, sending_rate: float, is_useful: bool,
        rtt_fluctuation_tolerance_ratio: float, rtt: float) -> None:
        if is_useful:
            self.num_useful_intervals += 1
        mi = MonitorInterval(self.mi_cnt, sending_rate, is_useful,
                             rtt_fluctuation_tolerance_ratio, rtt)
        self.q.append(mi)
        self.mi_cnt += 1

    def on_rtt_inflation_in_starting(self):
        self.q = []
        self.num_useful_intervals = 0
        self.num_available_intervals = 0
