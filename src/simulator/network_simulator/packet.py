from simulator.constants import BYTES_PER_PACKET
from simulator.network_simulator.constants import EVENT_TYPE_SEND
from simulator.network_simulator import sender


class Packet:
    """Packet event in simulator."""

    def __init__(self, ts: float, sender: "sender.Sender", pkt_id: int):
        self.ts = ts
        self.sent_time = ts
        self.dropped = False
        self.sender = sender
        self.event_type = EVENT_TYPE_SEND
        self.next_hop = 0
        self.pkt_id = pkt_id
        self.queue_delay = 0
        self.propagation_delay = 0
        self.pkt_size = BYTES_PER_PACKET # bytes

    def drop(self) -> None:
        """Mark packet as dropped."""
        self.dropped = True

    def add_propagation_delay(self, extra_delay: float) -> None:
        """Add to the propagation delay and add to the timestamp too."""
        self.propagation_delay += extra_delay
        self.ts += extra_delay

    def add_queue_delay(self, extra_delay: float) -> None:
        """Add to the queue delay and add to the timestamp too."""
        self.queue_delay += extra_delay
        self.ts += extra_delay

    @property
    def cur_latency(self) -> float:
        """Return Current latency experienced.

        Latency = propagation_delay + queue_delay
        """
        return self.queue_delay + self.propagation_delay

    @property
    def rtt(self) -> float:
        return self.cur_latency

    # override the comparison operator
    def __lt__(self, nxt):
        return self.ts < nxt.ts

    def debug_print(self):
        print("Event {}: ts={}, type={}".format(self.pkt_id, self.ts, self.event_type))
