import math
import random
from typing import Tuple

from simulator.network_simulator.constants import BITS_PER_BYTE, BYTES_PER_PACKET
from simulator.trace import Trace


class Link():

    def __init__(self, trace: Trace):
        self.trace = trace
        self.trace.reset()
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
        self.queue_size = self.trace.get_queue_size()
        self.pkt_in_queue = 0

    def get_cur_queue_delay(self, event_time: float) -> float:
        # pkt_in_queue_old = max(0, self.pkt_in_queue -
        #                         (event_time - self.queue_delay_update_time) *
        #                         self.get_bandwidth(event_time))
        self.pkt_in_queue = max(0, self.pkt_in_queue -
                                self.trace.get_avail_bits2send(self.queue_delay_update_time, event_time) / BITS_PER_BYTE / BYTES_PER_PACKET)
        # print('old pkt_in_queue', pkt_in_queue_old, 'new pkt_in_queue', pkt_in_queue, 'pkt_in_queue before change', self.pkt_in_queue)
        # self.pkt_in_queue = pkt_in_queue_old
        self.queue_delay_update_time = event_time
        # cur_queue_delay = math.ceil(
        #     self.pkt_in_queue) / self.get_bandwidth(event_time)

        # cur_queue_delay_old = self.pkt_in_queue / self.get_bandwidth(event_time) # cur_queue_delay is not accurate
        cur_queue_delay = self.trace.get_sending_t_usage(self.pkt_in_queue * BYTES_PER_PACKET * BITS_PER_BYTE, event_time)
        return cur_queue_delay

    def get_cur_latency(self, event_time: float) -> Tuple[float, float]:
        q_delay = self.get_cur_queue_delay(event_time)
        # print('queue delay: ', q_delay)
        return self.trace.get_delay(event_time) / 1000.0, q_delay

    def packet_enters_link(self, event_time: float) -> bool:
        if (random.random() < self.trace.get_loss_rate()):
            return False
        self.queue_delay = self.get_cur_queue_delay(event_time)
        extra_delay = 1.0 / self.get_bandwidth(event_time)
        # if 1 + math.ceil(self.pkt_in_queue) > self.queue_size:
        #     return False
        if 1 + self.pkt_in_queue > self.queue_size:
            return False
        self.queue_delay += extra_delay
        self.pkt_in_queue += 1
        return True

    def debug_print(self):
        print("Link:")
        print("Trace Avg Bandwidth: {:.3f}Mbps".format(self.get_bandwidth(0)))
        print("Trace Avg One-way Delay: {:.3f}ms".format(self.trace.get_delay(0)))
        print("Queue Delay: {:.3f}ms" % self.queue_delay)
        print("One Packet Queue Delay: %f" % (1.0 / self.get_bandwidth(0)))
        print("Queue size: {}" % self.queue_size)
        print("Loss: {:.3f}" % self.trace.get_loss_rate())
        raise NotImplementedError

    def reset(self):
        self.trace.reset()
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
        self.pkt_in_queue = 0

    def get_bandwidth(self, ts):
        return self.trace.get_bandwidth(ts) * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET
