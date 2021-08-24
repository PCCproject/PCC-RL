import csv
import os
from typing import Tuple

import numpy as np

from common.utils import pcc_aurora_reward
from plot_scripts.plot_packet_log import PacketLog
from simulator.network_simulator import packet
from simulator.network_simulator.constants import (
    BITS_PER_BYTE,
    BYTES_PER_PACKET,
)
from simulator.network_simulator.link import Link
from simulator.network_simulator.network import Network
from simulator.network_simulator.sender import Sender
from simulator.trace import Trace
from simulator.network_simulator.pcc.monitor_interval_queue import MonitorIntervalQueue


class VivaceLatencySender(Sender):
    def __init__(self, sender_id: int, dest: int):
        super().__init__(sender_id, dest)

        self.mi_queue = MonitorIntervalQueue()

    def on_packet_sent(self, pkt: "packet.Packet") -> None:
        return super().on_packet_sent(pkt)

    def on_packet_acked(self, pkt: "packet.Packet") -> None:
        return super().on_packet_acked(pkt)

    def on_packet_lost(self, pkt: "packet.Packet") -> None:
        return super().on_packet_lost(pkt)

    def create_new_interval(self) -> bool:
        """Determine whether to create a new MI."""
        if self.mi_queue.empty():
            return True
        # TODO:
        return False


class VivaceLatency:
    cc_name = 'vivace_latency'

    def __init__(self, record_pkt_log: bool = False):
        self.record_pkt_log = record_pkt_log

    def test(self, trace: Trace, save_dir: str) -> Tuple[float, float]:
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
        senders = [VivaceLatencySender(0, 0)]
        net = Network(senders, links, self.record_pkt_log)

        rewards = []
        start_rtt = trace.get_delay(0) * 2 / 1000
        run_dur = start_rtt
        if save_dir:
            writer = csv.writer(open(os.path.join(save_dir, '{}_simulation_log.csv'.format(
                self.cc_name)), 'w', 1), lineterminator='\n')
            writer.writerow(['timestamp', "send_rate", 'recv_rate', 'latency',
                             'loss', 'reward', "action", "bytes_sent",
                             "bytes_acked", "bytes_lost", "send_start_time",
                             "send_end_time", 'recv_start_time',
                             'recv_end_time', 'latency_increase',
                             "packet_size", 'bandwidth', "queue_delay",
                             'packet_in_queue', 'queue_size', 'cwnd',
                             'ssthresh', "rto", "packets_in_flight"])
        else:
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
            try:
                ssthresh = senders[0].ssthresh
            except:
                ssthresh = 0
            action = 0

            if save_dir and writer:
                writer.writerow([
                    net.get_cur_time(), send_rate, throughput, latency, loss,
                    reward, action, mi.bytes_sent, mi.bytes_acked, mi.bytes_lost,
                    mi.send_start, mi.send_end, mi.recv_start, mi.recv_end,
                    mi.get('latency increase'), mi.packet_size,
                    links[0].get_bandwidth(
                        net.get_cur_time()) * BYTES_PER_PACKET * BITS_PER_BYTE,
                    avg_queue_delay, links[0].pkt_in_queue, links[0].queue_size,
                    senders[0].cwnd, ssthresh, senders[0].rto,
                    senders[0].bytes_in_flight / BYTES_PER_PACKET])
            if senders[0].srtt:
                run_dur = senders[0].srtt
            should_stop = trace.is_finished(net.get_cur_time())
            if should_stop:
                break
        pkt_level_reward = 0
        if self.record_pkt_log and save_dir:
            with open(os.path.join(
                    save_dir, "{}_packet_log.csv".format(self.cc_name)), 'w', 1) as f:
                pkt_logger = csv.writer(f, lineterminator='\n')
                pkt_logger.writerow(['timestamp', 'packet_event_id',
                                     'event_type', 'bytes', 'cur_latency',
                                     'queue_delay', 'packet_in_queue',
                                     'sending_rate', 'bandwidth'])
                pkt_logger.writerows(net.pkt_log)
            pkt_log = PacketLog.from_log(net.pkt_log)
            pkt_level_reward = pkt_log.get_reward("", trace)
        return np.mean(rewards), pkt_level_reward
