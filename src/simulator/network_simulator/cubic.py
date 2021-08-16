import csv
import os

import numpy as np

from common.utils import pcc_aurora_reward
from simulator.network_simulator.constants import BITS_PER_BYTE, BYTES_PER_PACKET, MIN_CWND
from simulator.network_simulator.link import Link
from simulator.network_simulator.network import Network
from simulator.network_simulator.packet import Packet
from simulator.network_simulator.sender import Sender
from simulator.trace import Trace


class TCPCubicSender(Sender):

    # used by Cubic
    tcp_friendliness = 1
    fast_convergence = 1
    beta = 0.3
    C = 0.4

    # used by srtt
    ALPHA = 0.8
    BETA = 1.5

    def __init__(self, sender_id: int, dest: int, cwnd: int = 10):
        super().__init__(sender_id, dest)
        self.ssthresh = 200  # MAX_CWND
        self.pkt_loss_wait_time = 0
        self.cwnd = cwnd
        self.rto = 3  # retransmission timeout (seconds)
        # initialize rto to 3s for waiting SYC packets of TCP handshake
        self.srtt = None  # (self.ALPHA * self.srtt) + (1 - self.ALPHA) * rtt)
        self.timeout_cnt = 0
        self.timeout_mode = False
        self.min_latency = None

        self.cubic_reset()
        self.rtt_samples = []
        self.prev_rtt_samples = []

    def on_packet_sent(self, pkt: Packet):
        return super().on_packet_sent(pkt)

    def on_packet_acked(self, pkt: Packet):
        if not self.net:
            raise RuntimeError("network is not registered in sender.")
        super().on_packet_acked(pkt)
        rtt = pkt.cur_latency
        # Added by Zhengxu Xia
        if self.get_cur_time() > self.pkt_loss_wait_time:
            if self.dMin:
                self.dMin = min(self.dMin, rtt)
            else:
                self.dMin = rtt
            if self.cwnd <= self.ssthresh:  # in slow start region
                self.cwnd += 1
            else:  # in congestion avoidance or max bw probing region
                cnt = self.cubic_update()
                if self.cwnd_cnt > cnt:
                    self.cwnd += 1
                    self.cwnd_cnt = 0
                else:
                    self.cwnd_cnt += 1
        self.rtt_samples.append(rtt)
        # update RTO
        if self.srtt is None:
            self.srtt = rtt
        elif self.timeout_mode:
            self.srtt = rtt
            self.timeout_mode = False
        else:
            self.srtt = (self.ALPHA * self.srtt) + (1 - self.ALPHA) * rtt
        self.rto = max(1, min(self.BETA * self.srtt, 60))
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        if not self.rtt_samples and not self.prev_rtt_samples:
            raise RuntimeError(
                "prev_rtt_samples is empty. TCP session is not constructed successfully!")
        elif not self.rtt_samples:
            avg_sampled_rtt = float(np.mean(np.array(self.prev_rtt_samples)))
        else:
            avg_sampled_rtt = float(np.mean(np.array(self.rtt_samples)))
        self.rate = self.cwnd / avg_sampled_rtt

        # add packet into network
        self.send()

    def on_packet_lost(self, pkt: Packet) -> None:
        if not self.net:
            raise RuntimeError("network is not registered in sender.")
        super().on_packet_lost(pkt)
        rtt = pkt.cur_latency
        # print('packet_loss,', self.net.get_cur_time(), rtt, self.pkt_loss_wait_time)
        if self.get_cur_time() > self.pkt_loss_wait_time:
            self.pkt_loss_wait_time = self.get_cur_time() + rtt

            # print('packet_loss set wait time to', self.pkt_loss_wait_time,self.net.get_cur_time(), rtt)
            self.epoch_start = 0
            if self.cwnd < self.W_last_max and self.fast_convergence:
                self.W_last_max = self.cwnd * (2 - self.beta) / 2
            else:
                self.W_last_max = self.cwnd
            old_cwnd = self.cwnd
            self.cwnd = max(int(self.cwnd * (1 - self.beta)), 1)
            self.ssthresh = max(self.cwnd, MIN_CWND)
            # print("packet lost: cwnd change from", old_cwnd, "to", self.cwnd)
            if not self.rtt_samples and not self.prev_rtt_samples:
                # raise RuntimeError("prev_rtt_samples is empty. TCP session is not constructed successfully!")
                avg_sampled_rtt = rtt
            elif not self.rtt_samples:
                avg_sampled_rtt = float(
                    np.mean(np.array(self.prev_rtt_samples)))
            else:
                avg_sampled_rtt = float(np.mean(np.array(self.rtt_samples)))
            self.rate = self.cwnd / avg_sampled_rtt
        # else:
        #     self.cwnd += 1
            # self.pkt_loss_wait_time = int(self.cwnd)
            # self.pkt_loss_wait_time = 0 #int(self.cwnd)
            # print("{:.5f}\tloss\t{:.5f}\t{}\tpkt loss wait time={}".format(
            #     self.net.get_cur_time(), self.rate,
            #     self.timeout_cnt, self.pkt_loss_wait_time), file=sys.stderr,)

        # add packet into network
        self.send()

    def cubic_reset(self):
        self.W_last_max = 0
        self.epoch_start = 0
        self.origin_point = 0
        self.dMin = 0
        self.W_tcp = 0  # used in tcp friendliness
        self.K = 0
        # self.acked = 0 # TODO is this one used in Cubic?
        self.ack_cnt = 0

        # In standard, TCP cwnd_cnt is an additional state variable that tracks
        # the number of segments acked since the last cwnd increment; cwnd is
        # incremented only when cwnd_cnt > cwnd; then, cwnd_cnt is set to 0.
        # initialie to 0
        self.cwnd_cnt = 0
        self.pkt_loss_wait_time = 0
        self.timeout_mode = False

    def cubic_update(self):
        self.ack_cnt += 1
        # assume network current time is tcp_time_stamp
        assert self.net is not None
        tcp_time_stamp = self.get_cur_time()
        if self.epoch_start <= 0:
            self.epoch_start = tcp_time_stamp  # TODO: check the unit of time
            if self.cwnd < self.W_last_max:
                self.K = np.cbrt((self.W_last_max - self.cwnd)/self.C)
                self.origin_point = self.W_last_max
            else:
                self.K = 0
                self.origin_point = self.cwnd
            self.ack_cnt = 1
            self.W_tcp = self.cwnd
        t = tcp_time_stamp + self.dMin - self.epoch_start
        target = self.origin_point + self.C * (t - self.K)**3
        if target > self.cwnd:
            cnt = self.cwnd / (target - self.cwnd)
        else:
            cnt = 100 * self.cwnd
        # TODO: call friendliness
        return cnt

    def reset(self):
        super().reset()
        self.cubic_reset()
        self.min_latency = None
        # initialize rto to 3s for waiting SYC packets of TCP handshake
        self.rto = 3  # retransmission timeout (seconds)
        self.srtt = None
        self.timeout_cnt = 0

    def cubic_tcp_friendliness(self):
        raise NotImplementedError

    def send(self):
        if not self.net:
            raise RuntimeError("network is not registered in sender.")
        for _ in range(int(self.cwnd - self.bytes_in_flight / BYTES_PER_PACKET)):
            pkt = Packet(self.get_cur_time(), self, 0)
            self.net.add_packet(pkt)

    def can_send_packet(self) -> bool:
        return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd

    def schedule_send(self, first_pkt=False, on_ack=False):
        assert self.net, "network is not registered in sender."
        if first_pkt:
            for _ in range(self.cwnd):
                pkt = Packet(self.get_cur_time(), self, 0)
                self.net.add_packet(pkt)


class Cubic:
    cc_name = 'cubic'

    def __init__(self, save_dir: str, record_pkt_log: bool = False):
        self.save_dir = save_dir
        self.record_pkt_log = record_pkt_log

    def test(self, trace: Trace) -> float:

        links = [Link(trace), Link(trace)]
        senders = [TCPCubicSender(0, 0)]
        net = Network(senders, links, self.record_pkt_log)

        rewards = []
        start_rtt = trace.get_delay(0) * 2 / 1000
        run_dur = start_rtt
        writer = csv.writer(open(os.path.join(self.save_dir, '{}_simulation_log.csv'.format(
            self.cc_name)), 'w', 1), lineterminator='\n')
        writer.writerow(['timestamp', "send_rate", 'recv_rate', 'latency',
                             'loss', 'reward', "action", "bytes_sent",
                             "bytes_acked", "bytes_lost", "send_start_time",
                             "send_end_time", 'recv_start_time',
                             'recv_end_time', 'latency_increase',
                             "packet_size", 'bandwidth', "queue_delay",
                             'packet_in_queue', 'queue_size', 'cwnd',
                             'ssthresh', "rto"])

        while True:
            net.run(run_dur)
            mi = senders[0].get_run_data()

            throughput = mi.get("recv rate")  # bits/sec
            send_rate = mi.get("send rate")  # bits/sec
            latency = mi.get("avg latency")
            avg_queue_delay = mi.get("avg queue delay")
            loss = mi.get("loss ratio")

            reward = pcc_aurora_reward(
                throughput / BITS_PER_BYTE / BYTES_PER_PACKET, latency, loss,
                np.mean(trace.bandwidths) * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET)
            rewards.append(reward)
            ssthresh = senders[0].ssthresh
            action = 0

            writer.writerow([
                net.get_cur_time(), send_rate, throughput, latency, loss,
                reward, action, mi.bytes_sent, mi.bytes_acked, mi.bytes_lost,
                mi.send_start, mi.send_end, mi.recv_start, mi.recv_end,
                mi.get('latency increase'), mi.packet_size,
                links[0].get_bandwidth(net.get_cur_time()) * BYTES_PER_PACKET * BITS_PER_BYTE,
                avg_queue_delay, links[0].pkt_in_queue, links[0].queue_size,
                senders[0].cwnd, ssthresh, senders[0].rto])
            if senders[0].srtt:
                run_dur = senders[0].srtt
            should_stop = trace.is_finished(net.get_cur_time())
            if should_stop:
                break
        with open(os.path.join(self.save_dir, "{}_packet_log.csv".format(self.cc_name)), 'w', 1) as f:
            pkt_logger = csv.writer(f, lineterminator='\n')
            pkt_logger.writerow(['timestamp', 'packet_event_id', 'event_type',
                                 'bytes', 'cur_latency', 'queue_delay',
                                 'packet_in_queue', 'sending_rate', 'bandwidth'])
            pkt_logger.writerows(net.pkt_log)
        return 0
