from typing import Union
from simulator.network_simulator.pcc.monitor_interval import MonitorInterval


class MonitorIntervalQueue:

    def __init__(self) -> None:
        self.q = []

    def push(self, mi: MonitorInterval) -> None:
        self.q.append(mi)

    def pop(self) -> MonitorInterval:
        if self.empty():
            raise RuntimeError("MI queue is empty!")
        mi = self.q.pop(0)
        return mi

    def empty(self) -> bool:
        return len(self.q) == 0

    def current_mi(self) -> MonitorInterval:
        if self.empty():
            raise RuntimeError("MI queue is empty!")
        return self.q[-1]

    def front(self) -> MonitorInterval:
        if self.empty():
            raise RuntimeError("MI queue is empty!")
        return self.q[0]

    def on_pkt_sent(self, ts: float, pkt_id: int,
                    target_send_rate: float) -> None:
        if self.empty():
            raise RuntimeError("MI queue is empty!")
        current_mi = self.q[-1]
        current_mi.on_pkt_sent(ts, pkt_id, target_send_rate)

    def on_pkt_acked(self, ts: float, pkt_id: int, rtt: float,
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
