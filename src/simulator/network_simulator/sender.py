from typing import TypeVar

from common import sender_obs
from simulator.network_simulator import network, packet
from simulator.network_simulator.constants import BYTES_PER_PACKET


class Sender:

    SRTT_ALPHA = 1/8
    SRTT_BETA = 1/4

    def __init__(self, sender_id: int, dest: int):
        """Create a sender object.

        Args
            sender_id: id of sender device.
            rate: start sending rate. (Unit: packets per second).
            dest: id of destination device.
            features: a list of features used by RL model.
            cwnd: size of congestion window. (Unit: number of packets)
            history_len: length of history array. History array is used as the
                input of RL model.
        """
        self.sender_id = sender_id
        # variables to track in a MonitorInterval. Units: packet
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.rtt_samples = []

        self.rate = 0
        self.pkt_loss_wait_time = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.net = None
        self.dest = dest
        self.rto = -1
        self.ssthresh = 0

        self.srtt = None
        self.rttvar = None

    # def apply_rate_delta(self, delta):
    #     delta *= config.DELTA_SCALE
    #     if delta >= 0.0:
    #         self.set_rate(self.rate * (1.0 + delta))
    #     else:
    #         self.set_rate(self.rate / (1.0 - delta))
    #     # print("current rate {} after applying delta {}".format(self.rate, delta))
    #     # print("rate %f" % delta)
    #
    # def apply_cwnd_delta(self, delta):
    #     delta *= config.DELTA_SCALE
    #     #print("Applying delta %f" % delta)
    #     if delta >= 0.0:
    #         self.set_cwnd(self.cwnd * (1.0 + delta))
    #     else:
    #         self.set_cwnd(self.cwnd / (1.0 - delta))

    def can_send_packet(self) -> bool:
        return True

    def register_network(self, net: "network.Network") -> None:
        self.net = net

    def on_packet_sent(self, pkt: "packet.Packet") -> None:
        self.sent += 1
        self.bytes_in_flight += pkt.pkt_size

    def on_packet_acked(self, pkt: "packet.Packet") -> None:
        self.acked += 1
        assert self.bytes_in_flight >= pkt.pkt_size
        self.bytes_in_flight -= pkt.pkt_size
        if self.srtt is None and self.rttvar is None:
            self.srtt = pkt.rtt
            self.rttvar = pkt.rtt / 2
        elif self.srtt and self.rttvar:
            self.rttvar = (1 - self.SRTT_BETA) * self.rttvar + self.SRTT_BETA * abs(self.srtt - pkt.rtt)
            self.srtt = (1 - self.SRTT_ALPHA) * self.srtt + self.SRTT_ALPHA * pkt.rtt
        else:
            raise ValueError("srtt and rttvar shouldn't be None.")

        self.rtt_samples.append(pkt.rtt)


    def on_packet_lost(self, pkt: "packet.Packet") -> None:
        self.lost += 1
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
        rtt_samples = self.rtt_samples # if self.rtt_samples else self.prev_rtt_samples

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
    # def set_rate(self):
    #     raise NotImplementedError

    # def set_cwnd(self):
    #     raise NotImplementedError

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
        self.rate = 0
        self.bytes_in_flight = 0
        self.reset_obs()

    def timeout(self):
        # placeholder
        raise NotImplementedError


SenderType = TypeVar('SenderType', bound=Sender)
