import math
import random

from simulator.constants import BYTES_PER_PACKET
from simulator.trace import Trace


class Link():

    def __init__(self, trace: Trace):
        self.trace = trace
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
        self.queue_size = self.trace.get_queue_size()
        self.pkt_in_queue = 0

    def get_cur_queue_delay(self, event_time):
        # pkt_in_queue_old = max(0, self.pkt_in_queue -
        #                         (event_time - self.queue_delay_update_time) *
        #                         self.get_bandwidth(event_time))
        self.pkt_in_queue = max(0, self.pkt_in_queue -
                                self.trace.get_avail_bits2send(self.queue_delay_update_time, event_time) / 8 / BYTES_PER_PACKET)
        # print('old pkt_in_queue', pkt_in_queue_old, 'new pkt_in_queue', pkt_in_queue, 'pkt_in_queue before change', self.pkt_in_queue)
        # self.pkt_in_queue = pkt_in_queue_old
        self.queue_delay_update_time = event_time
        # cur_queue_delay = math.ceil(
        #     self.pkt_in_queue) / self.get_bandwidth(event_time)

        # cur_queue_delay_old = self.pkt_in_queue / self.get_bandwidth(event_time) # cur_queue_delay is not accurate
        cur_queue_delay = self.trace.get_sending_t_usage(self.pkt_in_queue * BYTES_PER_PACKET * 8, event_time)
        return cur_queue_delay

    def get_cur_latency(self, event_time):
        q_delay = self.get_cur_queue_delay(event_time)
        # print('queue delay: ', q_delay)
        return self.trace.get_delay(event_time) / 1000.0 + q_delay

    def packet_enters_link(self, event_time):
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

    def print_debug(self):
        print("Link:")
        print("Bandwidth: %f" % self.get_bandwidth(0))
        print("Delay: %f" % self.trace.get_delay(0))
        print("Queue Delay: %f" % self.queue_delay)
        print("One Packet Queue Delay: %f" % (1.0 / self.get_bandwidth(0)))
        print("Queue size: %d" % self.queue_size)
        print("Loss: %f" % self.trace.get_loss_rate())

    def reset(self):
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
        self.pkt_in_queue = 0

    def get_bandwidth(self, ts):
        return self.trace.get_bandwidth(ts) * 1e6 / 8 / BYTES_PER_PACKET
