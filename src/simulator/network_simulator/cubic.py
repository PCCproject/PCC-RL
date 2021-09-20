import csv
import os
import multiprocessing as mp
from typing import List, Tuple

import numpy as np

from common.utils import pcc_aurora_reward
from plot_scripts.plot_packet_log import PacketLog, plot
from plot_scripts.plot_time_series import plot as plot_simulation_log
from simulator.network_simulator.constants import (BITS_PER_BYTE, BYTES_PER_PACKET, MIN_CWND, TCP_INIT_CWND)
from simulator.network_simulator.link import Link
from simulator.network_simulator.network import Network
from simulator.network_simulator.packet import Packet
from simulator.network_simulator.sender import Sender
from simulator.trace import Trace


class TCPCubicSender(Sender):

    # used by Cubic
    tcp_friendliness = 0
    fast_convergence = 1
    beta = 0.3
    C = 0.4

    def __init__(self, sender_id: int, dest: int, cwnd: int = TCP_INIT_CWND):
        super().__init__(sender_id, dest)
        self.pkt_loss_wait_time = 0
        self.cwnd = cwnd
        self.timeout_cnt = 0

        self.cubic_reset()

    def on_packet_sent(self, pkt: Packet):
        return super().on_packet_sent(pkt)

    def on_packet_acked(self, pkt: Packet):
        if not self.net:
            raise RuntimeError("network is not registered in sender.")
        super().on_packet_acked(pkt)
        rtt = pkt.cur_latency
        # Added by Zhengxu Xia
        if self.get_cur_time() > self.pkt_loss_wait_time:
            self.in_fast_recovery_mode = False
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
        # add packet into network
        self.send()

    def on_packet_lost(self, pkt: Packet) -> None:
        if not self.net:
            raise RuntimeError("network is not registered in sender.")
        super().on_packet_lost(pkt)
        rtt = pkt.cur_latency
        # print('packet_loss,', self.net.get_cur_time(), rtt, self.pkt_loss_wait_time)
        if self.get_cur_time() > self.pkt_loss_wait_time:
            # : #
            if self.srtt is None:
                self.pkt_loss_wait_time = self.get_cur_time() + pkt.rtt
            else:
                self.pkt_loss_wait_time = self.get_cur_time() + self.srtt

            # print('packet_loss set wait time to', self.pkt_loss_wait_time,self.net.get_cur_time(), rtt)
            self.epoch_start = 0
            if self.cwnd < self.W_last_max and self.fast_convergence:
                self.W_last_max = self.cwnd * (2 - self.beta) / 2
            else:
                self.W_last_max = self.cwnd
            self.cwnd = max(int(self.cwnd * (1 - self.beta)), 1)
            self.ssthresh = max(self.cwnd, MIN_CWND)
            # print("packet lost: cwnd change from", old_cwnd, "to", self.cwnd)
        # else:
            # self.cwnd += 1
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
        self.ack_cnt = 0

        # In standard, TCP cwnd_cnt is an additional state variable that tracks
        # the number of segments acked since the last cwnd increment; cwnd is
        # incremented only when cwnd_cnt > cwnd; then, cwnd_cnt is set to 0.
        # initialie to 0
        self.cwnd_cnt = 0

    def cubic_update(self):
        self.ack_cnt += 1
        # assume network current time is tcp_time_stamp
        assert self.net is not None
        tcp_time_stamp = self.get_cur_time()
        if self.epoch_start <= 0:
            self.epoch_start = tcp_time_stamp
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
        # call friendliness
        if self.tcp_friendliness:
            cnt = self.cubic_tcp_friendliness(cnt)
        return cnt

    def reset(self):
        super().reset()
        self.cubic_reset()
        # initialize rto to 3s for waiting SYC packets of TCP handshake
        self.timeout_cnt = 0
        self.pkt_loss_wait_time = 0

    def timeout(self):
        # TODO: BUG!!!
        assert self.srtt
        self.bytes_in_flight -= BYTES_PER_PACKET

        self.cwnd = 1
        self.ssthresh = max(self.cwnd, MIN_CWND)
        # self.cubic_reset()
        self.pkt_loss_wait_time = self.get_cur_time() + self.srtt
        self.send()
        return
        # # if self.pkt_loss_wait_time <= 0:
        #     # self.timeout_cnt += 1
        #     # Refer to https://tools.ietf.org/html/rfc8312#section-4.7
        #     # self.ssthresh = max(int(self.bytes_in_flight / BYTES_PER_PACKET / 2), 2)
        #     self.ssthresh = self.cwnd * (1 - self.beta)
        #     old_cwnd = self.cwnd
        #     self.cwnd = 1
        #     self.cubic_reset()
        #     self.pkt_loss_wait_time = int(
        #         self.bytes_in_flight / BYTES_PER_PACKET)
        #     # self.timeout_mode = True
        #     # self.pkt_loss_wait_time = old_cwnd
        # # print('timeout rate', self.rate, self.cwnd)
        # # return True

    def cubic_tcp_friendliness(self, cnt):
        # TODO: buggy!
        self.W_tcp = self.W_tcp + 3 * self.beta / (2 - self.beta) * (self.ack_cnt / self.cwnd)

        if self.W_tcp > self.cwnd:
            max_cnt = self.cwnd / (self.W_tcp - self.cwnd)
            if cnt > max_cnt:
                cnt = max_cnt
        return cnt

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

    def __init__(self, record_pkt_log: bool = False):
        self.record_pkt_log = record_pkt_log

    def test(self, trace: Trace, save_dir: str, plot_flag: bool = False) -> Tuple[float, float]:
        """Test a network trace and return rewards.

        The 1st return value is the reward in Monitor Interval(MI) level and
        the length of MI is 1 srtt. The 2nd return value is the reward in
        packet level. It is computed by using throughput, average rtt, and
        loss rate in each 500ms bin of the packet log. The 2nd value will be 0
        if record_pkt_log flag is False.

        Args:
            trace: network trace.
            save_dir: where a MI level log will be saved if save_dir is a
                valid path. A packet level log will be saved if record_pkt_log
                flag is True and save_dir is a valid path.
        """

        links = [Link(trace), Link(trace)]
        senders = [TCPCubicSender(0, 0)]
        net = Network(senders, links, self.record_pkt_log)

        rewards = []
        start_rtt = trace.get_delay(0) * 2 / 1000
        run_dur = start_rtt
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            f_sim_log = open(os.path.join(save_dir, '{}_simulation_log.csv'.format(self.cc_name)), 'w', 1)
            writer = csv.writer(f_sim_log, lineterminator='\n')
            writer.writerow(['timestamp', "send_rate", 'recv_rate', 'latency',
                                 'loss', 'reward', "action", "bytes_sent",
                                 "bytes_acked", "bytes_lost", "send_start_time",
                                 "send_end_time", 'recv_start_time',
                                 'recv_end_time', 'latency_increase',
                                 "packet_size", 'bandwidth', "queue_delay",
                                 'packet_in_queue', 'queue_size', 'cwnd',
                                 'ssthresh', "rto", "packets_in_flight"])
        else:
            f_sim_log = None
            writer = None

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
            if save_dir and writer:
                writer.writerow([
                    net.get_cur_time(), send_rate, throughput, latency, loss,
                    reward, action, mi.bytes_sent, mi.bytes_acked, mi.bytes_lost,
                    mi.send_start, mi.send_end, mi.recv_start, mi.recv_end,
                    mi.get('latency increase'), mi.packet_size,
                    links[0].get_bandwidth(net.get_cur_time()) * BYTES_PER_PACKET * BITS_PER_BYTE,
                    avg_queue_delay, links[0].pkt_in_queue, links[0].queue_size,
                    senders[0].cwnd, ssthresh, senders[0].rto,
                    senders[0].bytes_in_flight / BYTES_PER_PACKET])
            if senders[0].srtt:
                run_dur = senders[0].srtt
            should_stop = trace.is_finished(net.get_cur_time())
            if should_stop:
                break
        if f_sim_log:
            f_sim_log.close()
        assert senders[0].last_ack_ts is not None and senders[0].first_ack_ts is not None
        assert senders[0].last_sent_ts is not None and senders[0].first_sent_ts is not None
        avg_sending_rate = senders[0].tot_sent / (senders[0].last_sent_ts - senders[0].first_sent_ts)
        tput = senders[0].tot_acked / (senders[0].last_ack_ts - senders[0].first_ack_ts)
        avg_lat = senders[0].cur_avg_latency
        loss = 1 - senders[0].tot_acked / senders[0].tot_sent
        pkt_level_reward = pcc_aurora_reward(tput, avg_lat,loss,
            avg_bw=trace.avg_bw * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET)
        if save_dir:
            with open(os.path.join(save_dir, "{}_summary.csv".format(self.cc_name)), 'w') as f:
                summary_writer = csv.writer(f, lineterminator='\n')
                summary_writer.writerow([
                    'trace_average_bandwidth', 'trace_average_latency',
                    'average_sending_rate', 'average_throughput',
                    'average_latency', 'loss_rate', 'mi_level_reward',
                    'pkt_level_reward'])
                summary_writer.writerow(
                    [trace.avg_bw, trace.avg_delay,
                     avg_sending_rate * BYTES_PER_PACKET * BITS_PER_BYTE / 1e6,
                     tput * BYTES_PER_PACKET * BITS_PER_BYTE / 1e6, avg_lat,
                     loss, np.mean(rewards), pkt_level_reward])
        if self.record_pkt_log and save_dir:
            with open(os.path.join(
                save_dir, "{}_packet_log.csv".format(self.cc_name)), 'w', 1) as f:
                pkt_logger = csv.writer(f, lineterminator='\n')
                pkt_logger.writerow(['timestamp', 'packet_event_id',
                                     'event_type', 'bytes', 'cur_latency',
                                     'queue_delay', 'packet_in_queue',
                                     'sending_rate', 'bandwidth', 'cwnd'])
                pkt_logger.writerows(net.pkt_log)
            # pkt_level_reward = pkt_log.get_reward("", trace)
        if self.record_pkt_log and plot_flag:
            pkt_log = PacketLog.from_log(net.pkt_log)
            plot(trace, pkt_log, save_dir, self.cc_name)
        if plot_flag and save_dir:
            plot_simulation_log(trace, os.path.join(save_dir, '{}_simulation_log.csv'.format(self.cc_name)), save_dir)
        return np.mean(rewards), pkt_level_reward

    def test_on_traces(self, traces: List[Trace], save_dirs: List[str],
                       plot_flag: bool = False, n_proc: int = 1):
        arguments = [(trace, save_dir, plot_flag) for trace, save_dir in zip(
            traces, save_dirs)]
        n_proc = n_proc
        with mp.Pool(processes=n_proc) as pool:
            return pool.starmap(self.test, arguments)
