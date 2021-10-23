import argparse
from bisect import bisect_right
import copy
import csv
import random
import os
from typing import List, Tuple, Union

import numpy as np
from common.utils import read_json_file, set_seed, write_json_file
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
    """

    def __init__(self, timestamps: Union[List[float], List[int]],
                 bandwidths: Union[List[int], List[float]],
                 delays: Union[List[int], List[float]], loss_rate: float,
                 queue_size: int, delay_noise: float = 0, offset=0):
        assert len(timestamps) == len(bandwidths)
        self.timestamps = timestamps
        if len(timestamps) >= 2:
            self.dt = timestamps[1] - timestamps[0]
        else:
            self.dt = 0.01

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

    @property
    def avg_bw(self) -> float:
        """Mean bandwidth in Mbps."""
        return np.mean(self.bandwidths)

    @property
    def min_delay(self) -> float:
        """Min one-way delay in ms."""
        return np.min(self.delays)

    @property
    def avg_delay(self) -> float:
        """Mean one-way delay in ms."""
        return np.mean(self.delays)

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
        # save trace details into a json file.
        data = {'timestamps': self.timestamps,
                'bandwidths': self.bandwidths,
                'delays': self.delays,
                'loss': self.loss_rate,
                'queue': self.queue_size,
                'delay_noise': self.delay_noise}
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
                                ms_per_bin: int = 500, front_offset: int = 0):
        flow = Flow(uplink_filename, ms_per_bin)
        downlink_filename = uplink_filename.replace('datalink', 'acklink')
        if downlink_filename and os.path.exists(downlink_filename):
            downlink = Flow(downlink_filename, ms_per_bin)
        else:
            raise FileNotFoundError
        delay = (np.min(flow.one_way_delay) + np.min(downlink.one_way_delay)) / 2
        timestamps = []
        bandwidths = []
        for ts, bw in zip(flow.throughput_timestamps, flow.throughput):
            if ts >= front_offset:
                timestamps.append(ts - front_offset)
                bandwidths.append(bw)

        # added to shift the trace 5 seconds
        # timestamps = [ts  - 5 for ts in flow.throughput_timestamps if ts >= 5]
        # tputs  = [tput for ts, tput in zip(flow.throughput_timestamps, flow.throughput) if ts >= 5]
        # tr = Trace(timestamps, tputs, [delay], loss, queue)

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


def generate_trace(duration_range: Tuple[float, float],
                   bandwidth_lower_bound_range: Tuple[float, float],
                   bandwidth_upper_bound_range: Tuple[float, float],
                   delay_range: Tuple[float, float],
                   loss_rate_range: Tuple[float, float],
                   queue_size_range: Tuple[float, float],
                   T_s_range: Union[Tuple[float, float], None] = None,
                   delay_noise_range: Union[Tuple[float, float], None] = None,
                   seed: Union[int, None] = None):
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
        delay_range[0], delay_range[1])

    queue_size = np.random.uniform(queue_size_range[0], queue_size_range[1])
    bdp = np.max(bandwidths) / BYTES_PER_PACKET / BITS_PER_BYTE * 1e6 * np.max(delays) * 2 / 1000
    queue_size = max(2, int(bdp * queue_size))

    ret_trace = Trace(timestamps, bandwidths, delays, loss_rate, queue_size, delay_noise)
    return ret_trace


def generate_traces(config_file: str, tot_trace_cnt: int, duration: int):
    config = read_json_file(config_file)
    traces = []
    weight_sum = 0
    for env_config in config:
        weight_sum += env_config['weight']
    assert round(weight_sum, 1) == 1.0

    for _ in range(tot_trace_cnt):
        trace = generate_trace_from_config_file(config_file, duration)
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


def generate_bw_series(prob_stay: float, T_s: float, cov: float, duration: float,
                       steps: int, min_bw: float, max_bw: float,
                       timestep: float = 1):
    """Generate a time-variant bandwidth series.

    Args:
        prob_stay: probability of staying in one state. Value range: [0, 1].
        T_s: how often the noise is changing. Value range: [0, inf)
        cov: maximum percentage of noise with resepect current bandwidth
            value. Value range: [0, 1].
        duration: trace duration in second.
        steps: number of steps.
        min_bw: minimum bandwidth in Mbps.
        max_bw: maximum bandwidth in Mbps.
        seed: numpy random seed.
        timestep: a bandwidth value every timestep seconds. Default: 1 second.

    """

    # equivalent to Pensieve's way of computing switch parameter
    coeffs = np.ones(steps - 1)
    coeffs[0] = -1
    switch_parameter = np.real(np.roots(coeffs)[0])
    """Generate a bandwidth series."""
    # get bandwidth levels (in Mbps)
    bw_states = []
    curr = min_bw
    for _ in range(0, steps):
        bw_states.append(curr)
        curr += (max_bw-min_bw)/(steps-1)

    # list of transition probabilities
    transition_probs = []
    # assume you can go steps-1 states away (we will normalize this to the
    # actual scenario)
    for z in range(1, steps-1):
        transition_probs.append(1/(switch_parameter**z))

    # takes a state and decides what the next state is
    current_state = np.random.randint(0, len(bw_states)-1)
    current_variance = cov * bw_states[current_state]
    ts = 0
    cnt = 0
    trace_time = []
    trace_bw = []
    noise = 0
    while ts < duration:
        # prints timestamp (in seconds) and throughput (in Mbits/s)
        if cnt <= 0:
            noise = np.random.normal(0, current_variance, 1)[0]
            cnt = T_s
        # the gaussian val is at least 0.1
        gaus_val = max(0.1, bw_states[current_state] + noise)
        trace_time.append(ts)
        trace_bw.append(gaus_val)
        cnt -= 1
        next_val = transition(current_state, prob_stay, bw_states,
                              transition_probs)
        if current_state != next_val:
            cnt = 0
        current_state = next_val
        current_variance = cov * bw_states[current_state]
        ts += timestep
    return trace_time, trace_bw


def transition(state, prob_stay, bw_states, transition_probs):
    """Hidden Markov State transition."""
    # variance_switch_prob, sigma_low, sigma_high,
    transition_prob = np.random.uniform()

    if transition_prob < prob_stay:  # stay in current state
        return state
    else:  # pick appropriate state!
        # next_state = state
        curr_pos = state
        # first find max distance that you can be from current state
        max_distance = max(curr_pos, len(bw_states)-1-curr_pos)
        # cut the transition probabilities to only have possible number of
        # steps
        curr_transition_probs = transition_probs[0:max_distance]
        trans_sum = sum(curr_transition_probs)
        normalized_trans = [x/trans_sum for x in curr_transition_probs]
        # generate a random number and see which bin it falls in to
        trans_switch_val = np.random.uniform()
        running_sum = 0
        num_switches = -1
        for ind in range(0, len(normalized_trans)):
            # this is the val
            if (trans_switch_val <= (normalized_trans[ind] + running_sum)):
                num_switches = ind
                break
            else:
                running_sum += normalized_trans[ind]

        # now check if there are multiple ways to move this many states away
        switch_up = curr_pos + num_switches
        switch_down = curr_pos - num_switches
        # can go either way
        if (switch_down >= 0 and switch_up <= (len(bw_states)-1)):
            x = np.random.uniform(0, 1, 1)
            if (x < 0.5):
                return switch_up
            else:
                return switch_down
        elif switch_down >= 0:  # switch down
            return switch_down
        else:  # switch up
            return switch_up


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Generate trace files.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to save the model.")
    parser.add_argument('--config-file', type=str, required=True,
                        help="config file")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    # parser.add_argument('--ntrace', type=int, required=True,
    #                     help='Number of trace files to be synthesized.')
    parser.add_argument('--time-variant-bw', action='store_true',
                        help='Generate time variant bandwidth if specified.')
    args, unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    # traces = generate_traces(args.config_file, args.ntrace, args.duration,
    #                          constant_bw=not args.time_variant_bw)
    conf = read_json_file(args.config_file)
    dim_vary = None
    for dim in conf:
        if len(conf[dim]) > 1:
            dim_vary = conf[dim]
        elif len(conf[dim]) == 1:
            pass
        else:
            raise RuntimeError

    assert dim_vary != None

    for value in dim_vary:
        os.makedirs(os.path.join(args.save_dir, str(value[1])), exist_ok=True)
        for i in range(10):
            tr = generate_trace(
                duration_range=conf['duration'][0] if len(
                    conf['duration']) == 1 else value,
                bandwidth_range=conf['bandwidth'][0] if len(
                    conf['bandwidth']) == 1 else (value[1], value[1]),
                delay_range=conf['delay'][0] if len(
                    conf['delay']) == 1 else value,
                loss_rate_range=conf['loss'][0] if len(
                    conf['loss']) == 1 else value,
                queue_size_range=conf['queue'][0] if len(
                    conf['queue']) == 1 else value,
                d_bw_range=conf['d_bw'][0] if len(conf['d_bw']) == 1 else value,
                d_delay_range=conf['d_delay'][0] if len(
                    conf['d_delay']) == 1 else value,
                T_s_range=conf['T_s'][0] if len(conf['T_s']) == 1 else value,
                delay_noise_range=conf['delay_noise'][0] if len(
                    conf['delay_nosie']) == 1 else value,
                constant_bw=False)
            tr.dump(os.path.join(args.save_dir, str(value[1]),
                                 'trace{:04d}.json'.format(i)))

# def generate_bw_delay_series(d_bw: float, d_delay, T_s: float,
#                              duration: float, min_tp: float, max_tp: float,
#                              min_delay: float, max_delay: float):
#     timestamps = []
#     bandwidths = []
#     delays = []
#     round_digit = 5
#     # if min_tp != max_tp:
#     #     target_bw = 0.6 #
#     # else:
#     target_bw = round(np.exp(float(np.random.uniform(
#             np.log(min_tp), np.log(max_tp), 1))), round_digit)
#
#     bw_val = round(np.exp(float(np.random.uniform(
#         np.log(min_tp), np.log(max_tp), 1))), round_digit)
#     if min_delay != max_delay:
#         target_delay = 5
#     else:
#         target_delay = round(float(np.random.uniform(min_delay, max_delay, 1)), round_digit)
#     delay_val = round(float(np.random.uniform(
#         min_delay, max_delay, 1)), round_digit)
#     ts = 0
#     bw_change_ts = 0
#     delay_change_ts = 0
#
#     # hard code default values
#     # if min_tp != max_tp and d_bw == 0:
#     #     target_bw = 0.6
#     # if min_delay != max_delay and d_delay == 0:
#     #     target_delay = 5
#     # handle T_s = 0 is wrong
#     # if T_s == 0:
#     #     target_delay = 5
#     #     target_bw = 0.6
#
#     while ts < duration:
#         if T_s !=0 and ts - bw_change_ts >= T_s:
#             new_bw = bw_val * (1 + float(np.random.normal(0, d_bw, 1)))
#             bw_val = max(min_tp, new_bw)
#             bw_change_ts = ts
#
#         if T_s != 0 and ts - delay_change_ts >= T_s:
#             new_delay = delay_val * \
#                 (1 + float(np.random.normal(0, d_delay, 1)))
#             delay_val = min(max(min_delay, new_delay), max_delay)
#             delay_change_ts = ts
#
#         ts = round(ts, round_digit)
#         timestamps.append(ts)
#         bandwidths.append(bw_val)
#         delays.append(delay_val)
#         ts += 0.1
#     timestamps.append(round(duration, round_digit))
#     bandwidths.append(bw_val)
#     delays.append(delay_val)
#
#     bw_mean = np.mean(np.array(bandwidths))
#     bandwidths = [round(float(val / bw_mean * target_bw), round_digit) for val in bandwidths]
#     delay_mean = np.mean(np.array(delays))
#     delays = [round(float(val / delay_mean * target_delay), round_digit) for val in delays]
#     return timestamps, bandwidths, delays


def generate_bw_delay_series(T_s: float, duration: float,
                             min_bw_lower_bnd: float, min_bw_upper_bnd: float,
                             max_bw_lower_bnd: float, max_bw_upper_bnd: float,
                             min_delay: float, max_delay: float)-> Tuple[List[float], List[float], List[float]]:
    timestamps = []
    bandwidths = []
    delays = []
    round_digit = 5
    min_bw_lower_bnd = round(min_bw_lower_bnd, round_digit)
    bw_upper_bnd =  round(np.exp(float(np.random.uniform(np.log(max_bw_lower_bnd), np.log(max_bw_upper_bnd), 1))), round_digit)
    assert min_bw_lower_bnd <= bw_upper_bnd, "{}, {}".format(min_bw_lower_bnd, bw_upper_bnd)
    bw_lower_bnd =  round(np.exp(float(np.random.uniform(np.log(min_bw_lower_bnd), np.log(min(min_bw_upper_bnd, bw_upper_bnd)), 1))), round_digit)
    # bw_val = round(np.exp(float(np.random.uniform(np.log(bw_lower_bnd), np.log(bw_upper_bnd), 1))), round_digit)
    bw_val = round(float(np.random.uniform(bw_lower_bnd, bw_upper_bnd, 1)), round_digit)
    delay_val = round(float(np.random.uniform(
        min_delay, max_delay, 1)), round_digit)
    ts = 0
    bw_change_ts = 0
    delay_change_ts = 0

    while ts < duration:
        if T_s !=0 and ts - bw_change_ts >= T_s:
            # TODO: how to change bw, uniform or logscale
            bw_val = float(np.random.uniform(bw_lower_bnd, bw_upper_bnd, 1))
            bw_change_ts = ts

        ts = round(ts, round_digit)
        timestamps.append(ts)
        bandwidths.append(bw_val)
        delays.append(delay_val)
        ts += 0.1
    timestamps.append(round(duration, round_digit))
    bandwidths.append(bw_val)
    delays.append(delay_val)

    return timestamps, bandwidths, delays

def generate_trace_from_config_file(config_file: str, duration: int = 30) -> Trace:
    config = read_json_file(config_file)
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
            # d_bw_min, d_bw_max = env_config['d_bw'] if 'd_bw' in env_config else (0, 0)
            # d_delay_min, d_delay_max = env_config['d_delay'] if 'd_delay' in env_config else (0, 0)
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

if __name__ == "__main__":
    main()
