import itertools
from typing import List, Tuple

import numpy as np
from common.utils import read_json_file


class Trace():
    """Trace object.

    timestamps and bandwidth should be at least list of one item if bandwidth
    is a constant. timestamps needs to contain the last timestamp of the trace
    to mark the duration of the trace.

    Args
        timestamps: trace timestamps in second.
        bandwidths: trace bandwidths in Mbps.
        delay: link delay in ms.
        queue: queue in packets.
    """

    def __init__(self, timestamps: List[float], bandwidths: List[float],
                 delay: float, loss_rate: float, queue_size: int):
        assert len(timestamps) == len(bandwidths)
        self.timestamps = timestamps
        self.bandwidths = bandwidths
        self.delay = delay
        self.loss_rate = loss_rate
        self.queue_size = queue_size

    def get_bandwidth(self, ts):
        """Return bandwidth(Mbps) at ts(second)."""
        if len(self.bandwidths) == 1:  # constant bandwidth
            return self.bandwidths[0]
        else:
            # TODO: support time-variant bandwidth
            raise NotImplementedError

    def get_delay(self):
        """Return link one-way delay(millisecond)."""
        return self.delay

    def get_loss_rate(self):
        """Return link loss rate."""
        return self.loss_rate

    def get_queue_size(self):
        return self.queue_size

    def is_finished(self, ts):
        """Return if trace is finished."""
        return ts >= self.timestamps[-1]

    def __str__(self):
        return ("Bandwidth: {:.3f}Mbps, Link delay: {:.3f}ms, "
                "Link loss: {:.3f}, Queue: {}packets".format(
                    self.bandwidths[0], self.delay, self.loss_rate,
                    self.queue_size))


def generate_trace(duration: float, bandwidth_range: Tuple[float, float],
                   delay_range: Tuple[float, float],
                   loss_rate_range: Tuple[float, float],
                   queue_size_range: Tuple[int, int],
                   constant_bandwidth=True):
    assert len(
        bandwidth_range) == 2 and bandwidth_range[0] <= bandwidth_range[1]
    assert len(delay_range) == 2 and delay_range[0] <= delay_range[1]
    assert len(
        loss_rate_range) == 2 and loss_rate_range[0] <= loss_rate_range[1]
    assert len(
        queue_size_range) == 2 and queue_size_range[0] <= queue_size_range[1] + 1

    delay = float(np.random.uniform(delay_range[0], delay_range[1], 1))
    loss_rate = float(np.random.uniform(
        loss_rate_range[0], loss_rate_range[1], 1))
    queue_size = int(np.random.randint(
        queue_size_range[0], queue_size_range[1]+1))

    if constant_bandwidth:
        bw = float(np.random.uniform(
            bandwidth_range[0], bandwidth_range[1], 1))
        return Trace([duration], [bw], delay, loss_rate, queue_size)

    raise NotImplementedError


def generate_traces(config_file: str, tot_trace_cnt: int, duration: int):
    config = read_json_file(config_file)
    bandwidth_ranges = config['train']['bandwidth']
    delay_ranges = config['train']['delay']
    loss_ranges = config['train']['loss']
    queue_ranges = config['train']['queue']
    traces = []

    for (bandwidth_range, delay_range, loss_range,
         queue_range) in itertools.product(bandwidth_ranges, delay_ranges,
                                           loss_ranges, queue_ranges):
        bw_min, bw_max, bw_prob = bandwidth_range
        delay_min, delay_max, delay_prob = delay_range
        loss_min, loss_max, loss_prob = loss_range
        queue_min, queue_max, queue_prob = queue_range
        trace_cnt = int(round(bw_prob * delay_prob *
                              loss_prob * queue_prob * tot_trace_cnt))
        for _ in range(trace_cnt):
            trace = generate_trace(duration, (bw_min, bw_max),
                                   (delay_min, delay_max),
                                   (loss_min, loss_max),
                                   (queue_min, queue_max))
            traces.append(trace)

    bandwidth_list = config['validation']['bandwidth']
    delay_list = config['validation']['delay']
    loss_list = config['validation']['loss']
    queue_list = config['validation']['queue']

    val_traces = []
    for bw, lat, loss, queue in itertools.product(bandwidth_list, delay_list,
                                                  loss_list, queue_list):
        val_traces.append(Trace([duration], [bw], lat, loss, queue))
    return traces, val_traces
