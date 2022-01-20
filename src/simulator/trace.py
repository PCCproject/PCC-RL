import argparse
from bisect import bisect_right
import copy
import csv
import random
import os
from typing import List, Tuple, Union

import numpy as np
from common.utils import read_json_file, set_seed, write_json_file, pcc_aurora_reward
from simulator.network_simulator.constants import BITS_PER_BYTE, BYTES_PER_PACKET
from simulator.pantheon_trace_parser.flow import Flow


class Trace():
    """Trace object.

    timestamps and bandwidth should be at least list of one item if bandwidth
    is a constant. timestamps needs to contain the last timestamp of the trace
    to mark the duration of the trace. bandwidhts and delays should share the
    same granularity.

    Args
        timestamps: trace timestamps in second.
        bandwidths: trace bandwidths in Mbps.
        delays: trace one-way delays in ms.
        loss_rate: uplink random packet loss rate.
        queue: queue in packets.
        delay_noise: maximum noise added to a packet in ms.
        bw_change_interval: bandwidth change interval in second.
    """

    def __init__(self, timestamps: Union[List[float], List[int]],
                 bandwidths: Union[List[int], List[float]],
                 delays: Union[List[int], List[float]], loss_rate: float,
                 queue_size: int, delay_noise: float = 0,
                 bw_change_interval: float = 0):
        assert len(timestamps) == len(bandwidths), \
                "len(timestamps)={}, len(bandwidths)={}".format(
                        len(timestamps), len(bandwidths))
        self.timestamps = timestamps
        if len(timestamps) >= 2:
            self.dt = timestamps[1] - timestamps[0]
        else:
            self.dt = 0.1

        self.bandwidths = [val if val >= 0.1 else 0.1 for val in bandwidths]
        self.delays = delays
        self.loss_rate = loss_rate
        self.queue_size = queue_size
        self.delay_noise = delay_noise
        self.noise = 0
        self.noise_change_ts = 0
        self.idx = 0  # track the position in the trace

        self.noise_timestamps = []
        self.noises = []
        self.noise_idx = 0
        self.return_noise = False
        self.bw_change_interval = bw_change_interval

    def real_trace_configs(self, normalized=False) -> List[float]:
        if normalized:
            return [(self.min_bw - 0.1) / (100 - 0.1),
                    (self.max_bw - 0.1) / (100 - 0.1),
                    (self.avg_delay - 0) / (200 - 2),
                    max((1 / self.bw_change_freq) / (30 - 0), 1)
                    if self.bw_change_freq > 0 else 1]
        return [self.min_bw, self.max_bw, self.avg_delay,
                1 / self.bw_change_freq]

    @property
    def bdp(self) -> float:
        return np.max(self.bandwidths) / BYTES_PER_PACKET / BITS_PER_BYTE * \
                1e6 * np.max(self.delays) * 2 / 1000

    @property
    def min_bw(self) -> float:
        """Min bandwidth in Mbps."""
        return np.min(self.bandwidths)

    @property
    def max_bw(self) -> float:
        """Max bandwidth in Mbps."""
        return np.max(self.bandwidths)

    @property
    def avg_bw(self) -> float:
        """Mean bandwidth in Mbps."""
        return np.mean(self.bandwidths)

    @property
    def std_bw(self) -> float:
        """Std of bandwidth in Mbps."""
        return np.std(self.bandwidths)

    @property
    def bw_change_freq(self) -> float:
        """Bandwidth change frequency in Hz."""
        avg_bw_per_sec = []
        t_start = self.timestamps[0]
        tot_bw = [self.bandwidths[0]]
        for ts, bw in zip(self.timestamps[1:], self.bandwidths[1:]):
            if (ts - t_start) < 0.5:
                tot_bw.append(bw)
            else:
                avg_bw_per_sec.append(np.mean(tot_bw))
                t_start = ts
                tot_bw = [bw]
        if tot_bw:
            avg_bw_per_sec.append(np.mean(tot_bw))
        change_cnt = 0
        for bw0, bw1 in zip(avg_bw_per_sec[:-1], avg_bw_per_sec[1:]):
            if (bw1 - bw0) / bw0 > 0.2: # value change greater than 20%
                change_cnt += 1

        # change_cnt = 0
        # for bw0, bw1 in zip(self.bandwidths[:-1], self.bandwidths[1:]):
        #     if (bw1 - bw0) / bw0 > 0.2: # value change greater than 20%
        #         change_cnt += 1
        return change_cnt / self.duration

    @property
    def duration(self) -> float:
        """Trace duration in second."""
        return self.timestamps[-1] - self.timestamps[0]

    @property
    def min_delay(self) -> float:
        """Min one-way delay in ms."""
        return np.min(np.array(self.delays))

    @property
    def avg_delay(self) -> float:
        """Mean one-way delay in ms."""
        return np.mean(np.array(self.delays))

    @property
    def optimal_reward(self):
        return pcc_aurora_reward(
                self.avg_bw * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET,
                self.avg_delay * 2 / 1000, self.loss_rate,
                self.avg_bw * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET)

    def get_next_ts(self) -> float:
        if self.idx + 1 < len(self.timestamps):
            return self.timestamps[self.idx+1]
        return 1e6

    def get_avail_bits2send(self, lo_ts: float, up_ts: float) -> float:
        lo_idx = bisect_right(self.timestamps, lo_ts) - 1
        up_idx = bisect_right(self.timestamps, up_ts) - 1
        avail_bits = sum(self.bandwidths[lo_idx: up_idx]) * 1e6 * self.dt
        avail_bits -= self.bandwidths[lo_idx] * 1e6 * (lo_ts - self.timestamps[lo_idx])
        avail_bits += self.bandwidths[up_idx] * 1e6 * (up_ts - self.timestamps[up_idx])
        return avail_bits

    def get_sending_t_usage(self, bits_2_send: float, ts: float) -> float:
        cur_idx = copy.copy(self.idx)
        t_used = 0

        while bits_2_send > 0:
            tmp_t_used = bits_2_send / (self.get_bandwidth(ts) * 1e6)
            if self.idx + 1 < len(self.timestamps) and tmp_t_used + ts > self.timestamps[self.idx + 1]:
                t_used += self.timestamps[self.idx + 1] - ts
                bits_2_send -= (self.timestamps[self.idx + 1] - ts) * (self.get_bandwidth(ts) * 1e6)
                ts = self.timestamps[self.idx + 1]
            else:
                t_used += tmp_t_used
                bits_2_send -= tmp_t_used * (self.get_bandwidth(ts) * 1e6)
                ts += tmp_t_used
            bits_2_send = round(bits_2_send, 9)

        self.idx = cur_idx # recover index
        return t_used

    def get_bandwidth(self, ts: float):
        """Return bandwidth(Mbps) at ts(second)."""
        # support time-variant bandwidth and constant bandwidth
        while self.idx + 1 < len(self.timestamps) and self.timestamps[self.idx + 1] <= ts:
            self.idx += 1
        if self.idx >= len(self.bandwidths):
            return self.bandwidths[-1]
        return self.bandwidths[self.idx]

    def get_delay(self, ts: float):
        """Return link one-way delay(millisecond) at ts(second)."""
        while self.idx + 1 < len(self.timestamps) and self.timestamps[self.idx + 1] <= ts:
            self.idx += 1
        if self.idx >= len(self.delays):
            return self.delays[-1]
        return self.delays[self.idx]

    def get_loss_rate(self):
        """Return link loss rate."""
        return self.loss_rate

    def get_queue_size(self):
        return self.queue_size

    def get_delay_noise(self, ts, cur_bw):
        # if self.delay_noise <= 0:
        #     return 0
        if ts - self.noise_change_ts > 1 / cur_bw:
        # self.noise = max(0, np.random.uniform(0, self.delay_noise, 1).item())
            self.noise = np.random.uniform(0, self.delay_noise, 1).item()
            self.noise_change_ts = ts
            ret =  self.noise
        else:
            ret = 0
        # print(ts, ret)
        return ret

    def get_delay_noise_replay(self, ts):
        while self.noise_idx + 1 < len(self.noise_timestamps) and self.noise_timestamps[self.noise_idx + 1] <= ts:
            self.noise_idx += 1
        if self.noise_idx >= len(self.noises):
            return self.noises[-1]
        return self.noises[self.noise_idx]

    def is_finished(self, ts: float):
        """Return if trace is finished."""
        return ts >= self.timestamps[-1]

    def __str__(self):
        return ("Timestamps: {}s,\nBandwidth: {}Mbps,\nLink delay: {}ms,\n"
                "Link loss: {:.3f}, Queue: {}packets".format(
                    self.timestamps, self.bandwidths, self.delays,
                    self.loss_rate, self.queue_size))

    def reset(self):
        self.idx = 0

    def dump(self, filename: str):
        """Save trace details into a json file."""
        data = {'timestamps': self.timestamps,
                'bandwidths': self.bandwidths,
                'delays': self.delays,
                'loss': self.loss_rate,
                'queue': self.queue_size,
                'delay_noise': self.delay_noise,
                'T_s': self.bw_change_interval}
        write_json_file(filename, data)

    @staticmethod
    def load_from_file(filename: str):
        trace_data = read_json_file(filename)
        tr = Trace(trace_data['timestamps'], trace_data['bandwidths'],
                   trace_data['delays'], trace_data['loss'],
                   trace_data['queue'], delay_noise=trace_data['delay_noise']
                   if 'delay_noise' in trace_data else 0)
        return tr

    @staticmethod
    def load_from_pantheon_file(uplink_filename: str, loss: float, queue: int,
                                ms_per_bin: int = 500, front_offset: float = 0,
                                wrap: bool = False):
        flow = Flow(uplink_filename, ms_per_bin)
        downlink_filename = uplink_filename.replace('datalink', 'acklink')
        if downlink_filename and os.path.exists(downlink_filename):
            downlink = Flow(downlink_filename, ms_per_bin)
        else:
            raise FileNotFoundError
        delay = (np.min(flow.one_way_delay) + np.min(downlink.one_way_delay)) / 2
        timestamps = []
        bandwidths = []
        wrapped_ts = []
        wrapped_bw = []
        for ts, bw in zip(flow.throughput_timestamps, flow.throughput):
            if ts >= front_offset:
                timestamps.append(ts - front_offset)
                bandwidths.append(bw)
            elif wrap:
                new_ts = flow.throughput_timestamps[-1] - front_offset + ms_per_bin / 1000 + ts
                if new_ts < 25:  # mimic the behavior in pantheon+mahimahi emulator.
                    wrapped_ts.append(new_ts)
                    wrapped_bw.append(bw)
        timestamps += wrapped_ts
        bandwidths += wrapped_bw

        tr = Trace(timestamps, bandwidths, [delay], loss, queue)
        return tr

    def convert_to_mahimahi_format(self):
        """
        timestamps: s
        bandwidths: Mbps
        """
        ms_series = []
        assert len(self.timestamps) == len(self.bandwidths)
        ms_t = 0
        for ts, next_ts, bw in zip(self.timestamps[0:-1], self.timestamps[1:], self.bandwidths[0:-1]):
            pkt_per_ms = bw * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET / 1000

            ms_cnt = 0
            pkt_cnt = 0
            while True:
                ms_cnt += 1
                ms_t += 1
                to_send = np.floor((ms_cnt * pkt_per_ms) - pkt_cnt)
                for _ in range(int(to_send)):
                    ms_series.append(ms_t)

                pkt_cnt += to_send

                if ms_cnt >= (next_ts - ts) * 1000:
                    break
        return ms_series

    def rotate_backward(self, offset: float):
        self.reset()
        timestamps = []
        bandwidths = []
        wrapped_ts = []
        wrapped_bw = []
        for ts, bw in zip(self.timestamps, self.bandwidths):
            if ts >= offset:
                timestamps.append(ts - offset)
                bandwidths.append(bw)
                wrapped_ts.append(self.timestamps[-1] - offset + self.dt + ts)
                wrapped_bw.append(bw)
        timestamps += wrapped_ts
        bandwidths += wrapped_bw
        self.timestamps = timestamps
        self.bandwidths = bandwidths


def generate_trace(duration_range: Tuple[float, float],
                   bandwidth_lower_bound_range: Tuple[float, float],
                   bandwidth_upper_bound_range: Tuple[float, float],
                   delay_range: Tuple[float, float],
                   loss_rate_range: Tuple[float, float],
                   queue_size_range: Tuple[float, float],
                   T_s_range: Union[Tuple[float, float], None] = None,
                   delay_noise_range: Union[Tuple[float, float], None] = None,
                   seed: Union[int, None] = None, dt: float = 0.1):
    """Generate trace for a network flow.

    Args:
        duration_range: duraiton range in second.
        bandwidth_range: link bandwidth range in Mbps.
        delay_range: link one-way propagation delay in ms.
        loss_rate_range: Uplink loss rate range.
        queue_size_range: queue size range in packets.
    """
    if seed:
        set_seed(seed)
    assert len(duration_range) == 2 and \
            duration_range[0] <= duration_range[1] and duration_range[0] > 0
    assert len(bandwidth_lower_bound_range) == 2 and \
            bandwidth_lower_bound_range[0] <= bandwidth_lower_bound_range[1] and bandwidth_lower_bound_range[0] > 0
    assert len(bandwidth_upper_bound_range) == 2 and \
            bandwidth_upper_bound_range[0] <= bandwidth_upper_bound_range[1] and bandwidth_upper_bound_range[0] > 0
    assert len(delay_range) == 2 and delay_range[0] <= delay_range[1] and \
            delay_range[0] > 0
    assert len(loss_rate_range) == 2 and \
            loss_rate_range[0] <= loss_rate_range[1] and loss_rate_range[0] >= 0

    loss_rate_exponent = float(np.random.uniform(np.log10(loss_rate_range[0]+1e-5), np.log10(loss_rate_range[1]+1e-5), 1))
    if loss_rate_exponent < -4:
        loss_rate = 0
    else:
        loss_rate = 10**loss_rate_exponent

    duration = float(np.random.uniform(
        duration_range[0], duration_range[1], 1))

    # use bandwidth generator.
    assert T_s_range is not None and len(
        T_s_range) == 2 and T_s_range[0] <= T_s_range[1]
    assert delay_noise_range is not None and len(
        delay_noise_range) == 2 and delay_noise_range[0] <= delay_noise_range[1]
    T_s = float(np.random.uniform(T_s_range[0], T_s_range[1], 1))
    delay_noise = float(np.random.uniform(delay_noise_range[0], delay_noise_range[1], 1))

    timestamps, bandwidths, delays = generate_bw_delay_series(
        T_s, duration, bandwidth_lower_bound_range[0], bandwidth_lower_bound_range[1],
        bandwidth_upper_bound_range[0], bandwidth_upper_bound_range[1],
        delay_range[0], delay_range[1], dt=dt)

    queue_size = np.random.uniform(queue_size_range[0], queue_size_range[1])
    bdp = np.max(bandwidths) / BYTES_PER_PACKET / BITS_PER_BYTE * 1e6 * np.max(delays) * 2 / 1000
    queue_size = max(2, int(bdp * queue_size))

    ret_trace = Trace(timestamps, bandwidths, delays, loss_rate, queue_size,
                      delay_noise, T_s)
    return ret_trace


def generate_traces(config_file: str, tot_trace_cnt: int, duration: int) -> List[Trace]:
    traces = []
    for _ in range(tot_trace_cnt):
        trace = generate_trace_from_config_file(config_file, duration)
        traces.append(trace)
    return traces


def generate_traces_from_config(config, tot_trace_cnt: int, duration: int) -> List[Trace]:
    traces = []
    for _ in range(tot_trace_cnt):
        trace = generate_trace_from_config(config, duration)
        traces.append(trace)
    return traces


def load_bandwidth_from_file(filename: str):
    timestamps = []
    bandwidths = []
    with open(filename, 'r') as f:
        csv_reader = csv.DictReader(f, delimiter=',')
        for row in csv_reader:
            timestamps.append(float(row['Timestamp']))
            bandwidths.append(float(row['Bandwidth']))

    return timestamps, bandwidths


def generate_bw_delay_series(T_s: float, duration: float,
                             min_bw_lower_bnd: float, min_bw_upper_bnd: float,
                             max_bw_lower_bnd: float, max_bw_upper_bnd: float,
                             min_delay: float, max_delay: float, dt: float=0.1) -> Tuple[List[float], List[float], List[float]]:
    timestamps = []
    bandwidths = []
    delays = []
    round_digit = 5
    min_bw_lower_bnd = round(min_bw_lower_bnd, round_digit)
    bw_upper_bnd =  round(np.exp(float(np.random.uniform(
        np.log(max_bw_lower_bnd), np.log(max_bw_upper_bnd), 1))), round_digit)
    assert min_bw_lower_bnd <= bw_upper_bnd, "{}, {}".format(
            min_bw_lower_bnd, bw_upper_bnd)
    bw_lower_bnd =  round(np.exp(float(np.random.uniform(
        np.log(min_bw_lower_bnd), np.log(min(min_bw_upper_bnd, bw_upper_bnd)), 1))), round_digit)
    # bw_val = round(np.exp(float(np.random.uniform(np.log(bw_lower_bnd), np.log(bw_upper_bnd), 1))), round_digit)
    bw_val = round(float(np.random.uniform(bw_lower_bnd, bw_upper_bnd, 1)), round_digit)
    delay_val = round(float(np.random.uniform(
        min_delay, max_delay, 1)), round_digit)
    ts = 0
    bw_change_ts = 0
    # delay_change_ts = 0

    while ts < duration:
        if T_s !=0 and ts - bw_change_ts >= T_s:
            # TODO: how to change bw, uniform or logscale
            bw_val = float(np.random.uniform(bw_lower_bnd, bw_upper_bnd, 1))
            bw_change_ts = ts

        ts = round(ts, round_digit)
        timestamps.append(ts)
        bandwidths.append(bw_val)
        delays.append(delay_val)
        ts += dt
    timestamps.append(round(duration, round_digit))
    bandwidths.append(bw_val)
    delays.append(delay_val)

    return timestamps, bandwidths, delays


def generate_trace_from_config_file(config_file: str, duration: int = 30) -> Trace:
    config = read_json_file(config_file)
    return generate_trace_from_config(config, duration)


def generate_trace_from_config(config, duration: int = 30) -> Trace:
    weight_sum = 0
    weights = []
    for env_config in config:
        weight_sum += env_config['weight']
        weights.append(env_config['weight'])
    assert round(weight_sum, 1) == 1.0
    indices_sorted = sorted(range(len(weights)), key=weights.__getitem__)
    weights_sorted = sorted(weights)
    weight_cumsums = np.cumsum(np.array(weights_sorted))

    rand_num = random.uniform(0, 1)

    for i, weight_cumsum in zip(indices_sorted, weight_cumsums):
        if rand_num <= float(weight_cumsum):
            env_config = config[i]
            bw_lower_bnd_min, bw_lower_bnd_max = env_config['bandwidth_lower_bound']
            bw_upper_bnd_min, bw_upper_bnd_max = env_config['bandwidth_upper_bound']
            delay_min, delay_max = env_config['delay']
            loss_min, loss_max = env_config['loss']
            queue_min, queue_max = env_config['queue']
            if 'duration' in env_config:
                duration_min, duration_max = env_config['duration']
            else:
                duration_min, duration_max = duration, duration

            # used by bandwidth generation
            delay_noise_min, delay_noise_max = env_config['delay_noise'] if 'delay_noise' in env_config else (0, 0)
            T_s_min, T_s_max = env_config['T_s'] if 'T_s' in env_config else (1, 1)
            return generate_trace((duration_min, duration_max),
                                  (bw_lower_bnd_min, bw_lower_bnd_max),
                                  (bw_upper_bnd_min, bw_upper_bnd_max),
                                  (delay_min, delay_max),
                                  (loss_min, loss_max),
                                  (queue_min, queue_max),
                                  (T_s_min, T_s_max),
                                  (delay_noise_min, delay_noise_max))
    raise ValueError("This line should never be reached.")


def generate_configs(config_file: str, n: int):
    config_range = read_json_file(config_file)[0]
    configs = []

    for _ in range(n):
        min_bw = 10**np.random.uniform(
                np.log10(config_range['bandwidth_lower_bound'][0]),
                np.log10(config_range['bandwidth_lower_bound'][1]), 1)[0]
        max_bw = 10**np.random.uniform(
                np.log10(config_range['bandwidth_upper_bound'][0]),
                np.log10(config_range['bandwidth_upper_bound'][1]), 1)[0]
        delay = np.random.uniform(config_range['delay'][0],
                                  config_range['delay'][1], 1)[0]
        queue = np.random.uniform(config_range['queue'][0],
                                  config_range['queue'][1], 1)[0]
        T_s = np.random.uniform(config_range['T_s'][0],
                                config_range['T_s'][1], 1)[0]
        loss_exponent = np.random.uniform(
                np.log10(config_range['loss'][0] + 1e-5),
                np.log10(config_range['loss'][1] + 1e-5), 1)[0]
        loss = 0 if loss_exponent < -4 else 10 ** loss_exponent
        configs.append([min_bw, max_bw, delay, queue, loss, T_s])

    return configs


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Training code.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to save the model.")
    parser.add_argument("--config-file", type=str, default=None,
                        help="A json file which contains a list of "
                        "randomization ranges with their probabilites.")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    assert args.count < 100000
    for i in range(args.count):
        trace = generate_trace_from_config_file(args.config_file)
        trace_file = os.path.join(args.save_dir, 'trace_{:05d}.json'.format(i))
        os.makedirs(args.save_dir, exist_ok=True)
        trace.dump(trace_file)


if __name__ == '__main__':
    main()
