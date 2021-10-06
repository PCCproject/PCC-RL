from typing import TypeVar

from common import sender_obs
from simulator.network_simulator import network, packet
from simulator.network_simulator.constants import BYTES_PER_PACKET


class Sender:
    """Base class of senders.

    srtt and retransmission reference: https://datatracker.ietf.org/doc/html/rfc6298
    """

    SRTT_ALPHA = 1 / 8
    SRTT_BETA = 1 / 4
    RTO_K = 4

    def __init__(self, sender_id: int, dest: int):
        """Create a sender object.

        Args
            sender_id: id of sender device.
            dest: id of destination device.
        """
        self.sender_id = sender_id
        # variables to track in a MonitorInterval. Units: packet
        self.sent = 0  # no. of packets
        self.acked = 0  # no. of packets
        self.lost = 0  # no. of packets
        self.rtt_samples = []
        self.queue_delay_samples = []

        # variables to track accross the connection session
        self.tot_sent = 0 # no. of packets
        self.tot_acked = 0 # no. of packets
        self.tot_lost = 0 # no. of packets
        self.cur_avg_latency = 0.0
        self.first_ack_ts = None
        self.last_ack_ts = None
        self.first_sent_ts = None
        self.last_sent_ts = None

        self.pacing_rate = 0  # bytes/s
        self.bytes_in_flight = 0  # bytes
        self.net = None
        self.dest = dest
        self.ssthresh = 80

        self.srtt = None
        self.rttvar = None
        # initialize rto to 3s for waiting SYC packets of TCP handshake
        self.rto = 3  # retransmission timeout (seconds)

        self.event_count = 0

    def can_send_packet(self) -> bool:
        return True

    def register_network(self, net: "network.Network") -> None:
        self.net = net

    def on_packet_sent(self, pkt: "packet.Packet") -> None:
        pkt.pkt_id = self.event_count
        self.event_count += 1
        self.sent += 1
        self.bytes_in_flight += pkt.pkt_size
        self.tot_sent += 1
        if self.first_sent_ts is None:
            self.first_sent_ts = pkt.ts
        self.last_sent_ts = pkt.ts

    def on_packet_acked(self, pkt: "packet.Packet") -> None:
        self.acked += 1
        self.cur_avg_latency = (self.cur_avg_latency * self.tot_acked + pkt.rtt) / (self.tot_acked + 1)
        self.tot_acked += 1
        if self.first_ack_ts is None:
            self.first_ack_ts = pkt.ts
        self.last_ack_ts = pkt.ts
        assert self.bytes_in_flight >= pkt.pkt_size
        self.bytes_in_flight -= pkt.pkt_size
        if self.srtt is None and self.rttvar is None:
            self.srtt = pkt.rtt
            self.rttvar = pkt.rtt / 2
            # RTO <- SRTT + max (G, K*RTTVAR) ignore G because clock granularity of modern os is tiny
            self.rto = max(1, min(self.srtt + self.RTO_K * self.rttvar, 60))
        elif self.srtt and self.rttvar:
            self.rttvar = (1 - self.SRTT_BETA) * self.rttvar + \
                self.SRTT_BETA * abs(self.srtt - pkt.rtt)
            self.srtt = (1 - self.SRTT_ALPHA) * self.srtt + \
                self.SRTT_ALPHA * pkt.rtt
            self.rto = max(1, min(self.srtt + self.RTO_K * self.rttvar, 60))
        else:
            raise ValueError("srtt and rttvar shouldn't be None.")

        self.rtt_samples.append(pkt.rtt)
        self.queue_delay_samples.append(pkt.queue_delay)

    def on_packet_lost(self, pkt: "packet.Packet") -> None:
        self.lost += 1
        self.tot_lost += 1
        assert self.bytes_in_flight >= pkt.pkt_size
        self.bytes_in_flight -= pkt.pkt_size

    def get_cur_time(self) -> float:
        assert self.net, "network is not registered in sender."
        return self.net.get_cur_time()

    def schedule_send(self, _) -> None:
        return

    def get_run_data(self):
        obs_end_time = self.get_cur_time()

        #obs_dur = obs_end_time - self.obs_start_time
        #print("Got %d acks in %f seconds" % (self.acked, obs_dur))
        #print("Sent %d packets in %f seconds" % (self.sent, obs_dur))
        #print("self.rate = %f" % self.rate)
        # print(self.acked, self.sent)
        # rtt_samples = self.rtt_samples # if self.rtt_samples else self.prev_rtt_samples

        return sender_obs.SenderMonitorInterval(
            self.sender_id,
            bytes_sent=self.sent * BYTES_PER_PACKET,
            bytes_acked=self.acked * BYTES_PER_PACKET,
            bytes_lost=self.lost * BYTES_PER_PACKET,
            send_start=self.obs_start_time,
            send_end=obs_end_time,
            recv_start=self.obs_start_time,
            recv_end=obs_end_time,
            rtt_samples=self.rtt_samples,
            queue_delay_samples=self.queue_delay_samples,
            packet_size=BYTES_PER_PACKET
        )

    # def print_debug(self):
    #     print("Sender:")
    #     print("Obs: %s" % str(self.get_obs()))
    #     print("Rate: %f" % self.rate)
    #     print("Sent: %d" % self.sent)
    #     print("Acked: %d" % self.acked)
    #     print("Lost: %d" % self.lost)
    #     print("Min Latency: %s" % str(self.min_latency))

    def reset_obs(self):
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.rtt_samples = []
        self.queue_delay_samples = []
        self.obs_start_time = self.get_cur_time()

    def reset(self):
        self.ssthresh = 80
        self.event_count = 0
        self.pacing_rate = 0
        self.bytes_in_flight = 0
        self.srtt = None
        self.rttvar = None
        self.rto = 3  # retransmission timeout (seconds)
        self.tot_sent = 0 # no. of packets
        self.tot_acked = 0 # no. of packets
        self.tot_lost = 0 # no. of packets
        self.cur_avg_latency = 0.0
        self.first_ack_ts = None
        self.last_ack_ts = None
        self.reset_obs()

    def timeout(self):
        # placeholder
        raise NotImplementedError

    def debug_print(self):
        pass


SenderType = TypeVar('SenderType', bound=Sender)
