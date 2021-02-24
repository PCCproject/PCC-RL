from simulator.network_simulator.constants import BYTES_PER_PACKET


class MonitorInterval():
    def __init__(self, start_time, end_time):
        """

        start_time: start time of the monitor interval.
        end_time: end time of monitor interval.
        """
        self.start_time = start_time  # interval start time
        self.end_time = end_time  # interval end time
        self.bytes_sent = 0  # bytes sent within this MI
        self.bytes_acked = 0  # bytes which have been acked
        self.bytes_lost = 0  # bytes which are considered as lost
        # self.n_packets_sent = 0  # packets senti witin this MI
        # self.n_packets_accounted_for = 0
        self.first_packet_send_time = 0
        self.last_packet_send_time = 0
        self.first_packet_ack_time = 0
        self.last_packet_ack_time = 0
        self.rtt_samples = []
        self.packet_size = BYTES_PER_PACKET

        # MI duraiton = 0.5 * RTT
        # 4 MI duration = 2 RTTs
        self.hard_stop_time = self.start_time + \
            (self.end_time - self.start_time) * 4

    def on_packet_sent(self, cur_time):
        if self.bytes_sent == 0:
            self.first_packet_send_time = cur_time
        self.bytes_sent += self.packet_size
        self.last_packet_send_time = cur_time
        # if self.bytes_sent == 8520:
        #     import ipdb
        #     ipdb.set_trace()

    def on_packet_acked(self, cur_time, rtt):
        if self.bytes_acked == 0:
            self.first_packet_ack_time = cur_time
        self.last_packet_ack_time = cur_time
        self.bytes_acked += BYTES_PER_PACKET
        self.rtt_samples.append(rtt)

    def on_packet_lost(self):
        self.bytes_lost += BYTES_PER_PACKET

    def is_finished(self, cur_time):
        return (cur_time >= self.end_time) and \
            ((self.bytes_sent == (self.bytes_acked + self.bytes_lost)))
             # or cur_time >= self.hard_stop_time)
