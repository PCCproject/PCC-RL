from typing import Union
from simulator.network_simulator.packet import Packet
from simulator.network_simulator.pcc.monitor_interval import MonitorInterval
from simulator.network_simulator.pcc.vivace.vivace_latency import VivaceLatencySender


class MonitorIntervalQueue:
    # Minimum number of reliable RTT samples per monitor interval.
    kMinReliableRtt = 4

    def __init__(self, sender: VivaceLatencySender) -> None:
        self.q = []
        self.sender = sender
        self.num_useful_intervals = 0
        self.num_available_intervals = 0
        self.mi_cnt = 0
        self.pending_acked_packets = []
        self.pending_avg_rtt = 0.0
        self.burst_flag = False
        self.pending_ack_interval = 0.0
        self.pending_event_time = 0.0
        self.pending_rtt = 0.0
        self.pending_avg_rtt = 0.0
        self.avg_interval_ratio = -1.0

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

    def on_packet_acked(self, pkt, ack_interval: float, latest_rtt: float,
            avg_rtt: float, min_rtt: float) -> None:
        if self.empty():
            raise RuntimeError("MI queue is empty!")
        self.num_available_intervals = 0

        if self.num_useful_intervals == 0:
            # Skip all the received packets if no intervals are useful.
            return
        has_invalid_utility = False

        for mi in self.q:
            if not mi.is_useful:
                continue
            for pending_acked_pkt in self.pending_acked_packets:
                if mi.contain_pkt(pending_acked_pkt):
                    if mi.bytes_acked == 0:
                        mi.rtt_on_monitor_start = self.pending_avg_rtt
                    mi.bytes_acked += pending_acked_pkt.pkt_size
                    is_reliable = False
                    if self.pending_ack_interval != 0:
                        interval_ratio = self.pending_ack_interval / ack_interval
                        if interval_ratio < 1.0:
                            interval_ratio = 1.0 / interval_ratio

                        if self.avg_interval_ratio < 0:
                            self.avg_interval_ratio = interval_ratio


                        if interval_ratio > 50.0 * self.avg_interval_ratio:
                            self.burst_flag = True
                        elif self.burst_flag:
                            if latest_rtt > self.pending_rtt and self.pending_rtt < self.pending_avg_rtt:
                                self.burst_flag = False
                        else:
                            is_reliable = True
                            mi.num_reliable_rtt += 1

                        self.avg_interval_ratio = self.avg_interval_ratio * 0.9 + interval_ratio * 0.1
                    is_reliable_for_gradient_calculation = False
                    if is_reliable:
                        is_reliable_for_gradient_calculation = True
                        mi.num_reliable_rtt_for_gradient_calculation += 1


                    mi.packet_rtt_samples.push_back(PacketRttSample(
                        pending_acked_pkt.packet_number, self.pending_rtt, self.pending_event_time,
                        is_reliable, is_reliable_for_gradient_calculation))
                    if mi.num_reliable_rtt >= self.kMinReliableRtt:
                        mi.has_enough_reliable_rtt = True
            if self.is_utility_available(mi):
                mi.rtt_on_monitor_end = avg_rtt
                mi.min_rtt = min_rtt
                has_invalid_utility = self.has_invalid_utility(mi)
                if has_invalid_utility:
                    break

                self.num_available_intervals += 1
                assert self.num_available_intervals <= self.num_useful_intervals
        self.pending_acked_packets = []
        self.pending_acked_packets.append(pkt)
        self.pending_rtt = latest_rtt
        self.pending_avg_rtt = avg_rtt
        self.pending_ack_interval = ack_interval
        self.pending_event_time = pkt.ts

        if self.num_useful_intervals > self.num_available_intervals and not has_invalid_utility:
            return


        if not has_invalid_utility:
            assert self.num_useful_intervals > 0

            useful_intervals = []
            for mi in self.q:
                if not mi.is_useful:
                    continue

                useful_intervals.append(mi)

            assert self.num_available_intervals == len(useful_intervals)

            self.sender.on_utility_available(useful_intervals, pkt.ts)

        # Remove MonitorIntervals from the head of the queue,
        # until all useful intervals are removed.
        while self.num_useful_intervals > 0:
            if self.q[0].is_useful:
                self.num_useful_intervals -= 1
            self.q.pop(0)

        self.num_available_intervals = 0


    def on_packet_lost(self, ts, pkt_id) -> None:
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

    def is_utility_available(self, interval):
        return (interval.has_enough_reliable_rtt and
          interval.bytes_acked + interval.bytes_lost == interval.bytes_sent)


    def has_invalid_utility(self, interval):
        return interval.first_packet_sent_time == interval.last_packet_sent_time

