import random


class Link():

    def __init__(self, bandwidth, delay, queue_size, loss_rate):
        self.bw = float(bandwidth)
        self.dl = delay
        self.lr = loss_rate
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
        self.max_queue_delay = queue_size / self.bw
        self.queue_size = queue_size

    def get_cur_queue_delay(self, event_time):
        cur_queue_delay = max(0.0,
                              self.queue_delay - (event_time -
                                                  self.queue_delay_update_time))
        # print('Event Time: {}s, queue_delay: {}s Queue_delay_update_time:
        # {}s, cur_queue_delay: {}s, max_queue_delay: {}s, bw: {}, queue size:
        # {}'.format( event_time, self.queue_delay,
        # self.queue_delay_update_time, cur_queue_delay, self.max_queue_delay,
        # self.bw, self.queue_size))
        return cur_queue_delay

    def get_cur_latency(self, event_time):
        q_delay = self.get_cur_queue_delay(event_time)
        # print('queue delay: ', q_delay)
        return self.dl + q_delay

    def packet_enters_link(self, event_time):
        if (random.random() < self.lr):
            return False
        self.queue_delay = self.get_cur_queue_delay(event_time)
        self.queue_delay_update_time = event_time
        extra_delay = 1.0 / self.bw
        # print("Extra delay:{}, Current delay: {}, Max delay:
        # {}".format(extra_delay, self.queue_delay, self.max_queue_delay))
        if extra_delay + self.queue_delay > self.max_queue_delay:
            # print("\tDrop!")
            return False
        self.queue_delay += extra_delay
        # print("\tNew delay = {}".format(self.queue_delay))
        return True

    def print_debug(self):
        print("Link:")
        print("Bandwidth: %f" % self.bw)
        print("Delay: %f" % self.dl)
        print("Queue Delay: %f" % self.queue_delay)
        print("Max Queue Delay: %f" % self.max_queue_delay)
        print("One Packet Queue Delay: %f" % (1.0 / self.bw))

    def reset(self):
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0

    def __str__(self):
        return "Link bw={}pkts, delay={}s, loss={}, queue={}pkts".format(
            self.bw, self.dl, self.lr, self.queue_size)
