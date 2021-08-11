from typing import TypeVar

from simulator.network_simulator import network, packet


class Sender:

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

        self.rate = 0
        self.pkt_loss_wait_time = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_sample = 0
        self.net = None
        self.dest = dest
        self.rto = -1

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
        self.bytes_in_flight -= pkt.pkt_size

    def on_packet_lost(self, pkt: "packet.Packet") -> None:
        self.lost += 1
        self.bytes_in_flight -= pkt.pkt_size

    def get_cur_time(self) -> float:
        assert self.net, "network is not registered in sender."
        return self.net.get_cur_time()

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

    def reset(self):
        self.rate = 0
        self.bytes_in_flight = 0
        self.sent = 0
        self.acked = 0
        self.lost = 0

    def timeout(self):
        # placeholder
        raise NotImplementedError


SenderType = TypeVar('SenderType', bound=Sender)
