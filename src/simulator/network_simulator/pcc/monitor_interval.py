import sys
from typing import List

import numpy as np

from simulator.network_simulator.constants import BYTES_PER_PACKET, START_SENDING_RATE


class MonitorInterval:

    def __init__(self, mi_id: int, sender_id: int,
                 send_rate: float = START_SENDING_RATE,
                 end_time: float = 0) -> None:
        self.send_rate = send_rate
        self.end_time = end_time
        self.features = {}
        self.sender_id = sender_id
        self.bytes_sent = 0
        self.bytes_acked = 0
        self.bytes_lost = 0
        self.send_start = 0
        self.send_end = 0
        self.recv_start = 0
        self.recv_end = 0
        self.rtt_samples = []
        self.queue_delay_samples = []
        self.packet_size = BYTES_PER_PACKET

        self.n_pkts_sent = 0
        self.n_pkts_accounted_for = 0
        self.first_pkt_id = -1  # first pkt sent
        self.last_pkt_id = -1  # last pkt sent

        self.mi_id = mi_id

    def debug_print(self) -> None:
        print("mi_id={}, end_time={:.3f}s, send_rate={:.0f}bps, "
              "bytes_sent={:.0f}bytes, bytes_acked={:.0f}bytes, "
              "bytes_lost={:.0f}bytes, send_start={:.3f}s, send_end={:.3f}s, "
              "recv_start={:.3f}s, recv_end={:.3f}s, n_pkts_sent={}, "
              "n_pkts_accounted_for={}, send_rate={:.0f}mbps, "
              "recv_rate={:.0f}bps, avg_latency={:.3f}s, "
              "loss={:.3f}, rtt_samples={}".format(
                  self.mi_id, self.end_time,
                  self.send_rate * BYTES_PER_PACKET * 8, self.bytes_sent,
                  self.bytes_acked, self.bytes_lost, self.send_start,
                  self.send_end, self.recv_start, self.recv_end,
                  self.n_pkts_sent, self.n_pkts_accounted_for,
                  self.get('send rate'), self.get('recv rate'),
                  self.get('avg latency'), self.get('loss ratio'),
                  self.rtt_samples), file=sys.stderr)

    def as_array(self, features):
        return np.array([self.get(f) / MonitorIntervalMetric.get_by_name(f).scale for f in features])

    def get(self, feature):
        if feature in self.features.keys():
            return self.features[feature]
        else:
            result = MonitorIntervalMetric.eval_by_name(feature, self)
            self.features[feature] = result
            return result

    def on_pkt_sent(self, ts: float, pkt_id: int,
                    target_send_rate: float) -> None:
        if self.n_pkts_sent == 0:
            self.send_start = ts
            self.first_pkt_id = pkt_id
        self.send_end = ts
        self.last_pkt_id = pkt_id
        self.n_pkts_sent += 1
        self.bytes_sent += BYTES_PER_PACKET
        self.send_rate = target_send_rate

    def on_pkt_acked(self, ts: float, pkt_id: int, rtt: float,
                     queue_delay:float) -> None:
        if self.contain_pkt(pkt_id):
            self.bytes_acked += BYTES_PER_PACKET
            self.rtt_samples.append(round(rtt, 9))
            self.queue_delay_samples.append(queue_delay)
            self.n_pkts_accounted_for += 1  # Different from emulation
            self.recv_end = ts

        if pkt_id >= self.first_pkt_id and self.recv_start == 0:
            self.recv_start = ts
        if pkt_id >= self.last_pkt_id and self.recv_end == 0:
            self.recv_end = ts

    def on_pkt_lost(self, ts: float, pkt_id: int) -> None:
        if self.contain_pkt(pkt_id):
            self.bytes_lost += BYTES_PER_PACKET
            self.n_pkts_accounted_for += 1  # Different from emulation

        if pkt_id >= self.first_pkt_id and self.recv_start == 0:
            self.recv_start = ts
        if pkt_id >= self.last_pkt_id and self.recv_end == 0:
            self.recv_end = ts

    def all_pkts_sent(self, ts: float) -> bool:
        return ts >= self.end_time

    def all_pkts_accounted_for(self, ts: float) -> bool:
        return self.all_pkts_sent(ts) and (self.n_pkts_accounted_for == self.n_pkts_sent)

    def contain_pkt(self, pkt_id: int) -> bool:
        """Check if the MI sent the packet or not."""
        return (pkt_id >= self.first_pkt_id) and (pkt_id <= self.last_pkt_id)


class MonitorIntervalHistory():
    def __init__(self, length: int, features: List[str], sender_id: int) -> None:
        self.features = features
        self.values = []
        self.sender_id = sender_id
        for i in range(length):
            self.values.append(MonitorInterval(i, self.sender_id))

    def step(self, new_mi) -> None:
        self.values.pop(0)
        self.values.append(new_mi)

    def as_array(self):
        arrays = []
        for mi in self.values:
            arrays.append(mi.as_array(self.features))
        arrays = np.array(arrays).flatten()
        return arrays

    def get_latest_mi(self) -> MonitorInterval:
        return self.values[-1]


class MonitorIntervalMetric():
    _all_metrics = {}

    def __init__(self, name, func, min_val, max_val, scale=1.0):
        self.name = name
        self.func = func
        self.min_val = min_val
        self.max_val = max_val
        self.scale = scale
        MonitorIntervalMetric._all_metrics[name] = self

    def eval(self, mi):
        return self.func(mi)

    @staticmethod
    def eval_by_name(name, mi):
        return MonitorIntervalMetric._all_metrics[name].eval(mi)

    @staticmethod
    def get_by_name(name):
        return MonitorIntervalMetric._all_metrics[name]


def get_min_obs_vector(feature_names):
    result = []
    for feature_name in feature_names:
        feature = MonitorIntervalMetric.get_by_name(feature_name)
        result.append(feature.min_val)
    return np.array(result)


def get_max_obs_vector(feature_names):
    result = []
    for feature_name in feature_names:
        feature = MonitorIntervalMetric.get_by_name(feature_name)
        result.append(feature.max_val)
    return np.array(result)


def _mi_metric_recv_rate(mi):
    dur = mi.get("recv dur")
    if dur > 0.0:
        # if 8.0 * (mi.bytes_acked - mi.packet_size) / dur < 0:
        #     return 0
        # return 8.0 * (mi.bytes_acked - mi.packet_size) / dur
        return 8.0 * (mi.bytes_acked - BYTES_PER_PACKET) / dur
    return 0.0


def _mi_metric_recv_dur(mi):
    return mi.recv_end - mi.recv_start


def _mi_metric_avg_latency(mi):
    if len(mi.rtt_samples) > 0:
        return np.mean(mi.rtt_samples)
    return 0.0


def _mi_metric_avg_queue_delay(mi):
    if len(mi.queue_delay_samples) > 0:
        return np.mean(mi.queue_delay_samples)
    return 0.0


def _mi_metric_send_rate(mi):
    dur = mi.get("send dur")
    if dur > 0.0:
        return 8.0 * mi.bytes_sent / dur
    return 0.0


def _mi_metric_send_dur(mi):
    return mi.send_end - mi.send_start + 1 / mi.send_rate


def _mi_metric_loss_ratio(mi):
    if mi.bytes_lost + mi.bytes_acked > 0:
        return mi.bytes_lost / (mi.bytes_lost + mi.bytes_acked)
    return 0.0


def _mi_metric_latency_increase(mi):
    half = int(len(mi.rtt_samples) / 2)
    if half >= 1:
        return np.mean(mi.rtt_samples[half:]) - np.mean(mi.rtt_samples[:half])
    return 0.0


def _mi_metric_ack_latency_inflation(mi):
    dur = mi.get("recv dur")
    latency_increase = mi.get("latency increase")
    if dur > 0.0:
        return latency_increase / dur
    return 0.0


def _mi_metric_sent_latency_inflation(mi):
    dur = mi.get("send dur")
    latency_increase = mi.get("latency increase")
    if dur > 0.0:
        return latency_increase / dur
    return 0.0


_conn_min_latencies = {}


def _mi_metric_conn_min_latency(mi):
    # latency = mi.get("avg latency")
    # change min latency from min average latency of a MI into min latency of
    # a packet
    if len(mi.rtt_samples) > 0:
        latency = min(mi.rtt_samples)
    else:
        latency = 0
    if mi.sender_id in _conn_min_latencies.keys():
        prev_min = _conn_min_latencies[mi.sender_id]
        if latency == 0.0:
            return prev_min
        else:
            if latency < prev_min:
                _conn_min_latencies[mi.sender_id] = latency
                return latency
            else:
                return prev_min
    else:
        if latency > 0.0:
            _conn_min_latencies[mi.sender_id] = latency
            return latency
        else:
            return 0.0


def _mi_metric_send_ratio(mi):
    thpt = mi.get("recv rate")
    send_rate = mi.get("send rate")
    if (thpt > 0.0) and (send_rate < 1000.0 * thpt):
        return send_rate / thpt
    # elif thpt == 0:
    #     return 2 #send_rate / 0.1
    return 1.0


def _mi_metric_latency_ratio(mi):
    min_lat = mi.get("conn min latency")
    cur_lat = mi.get("avg latency")
    if min_lat > 0.0:
        return cur_lat / min_lat
    return 1.0


SENDER_MI_METRICS = [
    MonitorIntervalMetric(
        "send rate", _mi_metric_send_rate, 0.0, 1e9, 1e7),
    MonitorIntervalMetric(
        "recv rate", _mi_metric_recv_rate, 0.0, 1e9, 1e7),
    MonitorIntervalMetric("recv dur", _mi_metric_recv_dur, 0.0, 100.0),
    MonitorIntervalMetric("send dur", _mi_metric_send_dur, 0.0, 100.0),
    MonitorIntervalMetric(
        "avg latency", _mi_metric_avg_latency, 0.0, 100.0),
    MonitorIntervalMetric(
        "avg queue delay", _mi_metric_avg_queue_delay, 0.0, 100.0),
    MonitorIntervalMetric("loss ratio", _mi_metric_loss_ratio, 0.0, 1.0),
    MonitorIntervalMetric(
        "ack latency inflation", _mi_metric_ack_latency_inflation, -1.0, 10.0),
    MonitorIntervalMetric(
        "sent latency inflation", _mi_metric_sent_latency_inflation, -1.0, 10.0),
    MonitorIntervalMetric(
        "conn min latency", _mi_metric_conn_min_latency, 0.0, 100.0),
    MonitorIntervalMetric(
        "latency increase", _mi_metric_latency_increase, 0.0, 100.0),
    MonitorIntervalMetric(
        "latency ratio", _mi_metric_latency_ratio, 1.0, 10000.0),
    MonitorIntervalMetric(
        "send ratio", _mi_metric_send_ratio, 0.0, 1000.0)
]
