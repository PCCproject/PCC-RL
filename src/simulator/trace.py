import csv
from typing import List, Tuple, Union

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
        self.idx = 0  # track the position in the trace

    def get_bandwidth(self, ts):
        """Return bandwidth(Mbps) at ts(second)."""
        # support time-variant bandwidth and constant bandwidth
        while self.idx + 1 < len(self.timestamps) and self.timestamps[self.idx + 1] <= ts:
            self.idx += 1
        return self.bandwidths[self.idx]

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
                    self.bandwidths[self.idx], self.delay, self.loss_rate,
                    self.queue_size))

    def reset(self):
        self.idx = 0

    def dump(self, filename):
        # TODO: save trace details into a json file.
        raise NotImplementedError


def generate_trace(duration_range: Tuple[float, float],
                   bandwidth_range: Tuple[float, float],
                   delay_range: Tuple[float, float],
                   loss_rate_range: Tuple[float, float],
                   queue_size_range: Tuple[int, int],
                   T_l_range: Union[Tuple[float, float], None] = None,
                   T_s_range: Union[Tuple[float, float], None] = None,
                   cov_range: Union[Tuple[float, float], None] = None,
                   steps_range: Union[Tuple[int, int], None] = None,
                   timestep_range: Union[Tuple[float, float], None] = None,
                   bandwidth_file: str = "",
                   constant_bw: bool = True):
    """Generate trace for a network flow.

    Args:
        duration_range: duraiton range in second.
        bandwidth_range: link bandwidth range in Mbps.
        delay_range: link one-way propagation delay in ms.
        loss_rate_range: Uplink loss rate range.
        queue_size_range: queue size range in packets.
    """
    assert len(
        duration_range) == 2 and duration_range[0] <= duration_range[1]
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
    # queue_size = int(np.random.randint(
    #     queue_size_range[0], queue_size_range[1]+1))

    queue_size = int(np.exp(np.random.uniform(
        np.log(queue_size_range[0]),
        np.log(queue_size_range[1]+1), 1)))

    if bandwidth_file:
        timestamps, bandwidths = load_bandwidth_from_file(bandwidth_file)
        return Trace(timestamps, bandwidths, delay, loss_rate, queue_size)

    duration = float(np.random.uniform(
        duration_range[0], duration_range[1], 1))
    if constant_bw:
        bw = float(np.random.uniform(
            bandwidth_range[0], bandwidth_range[1], 1))
        ret_trace = Trace([duration], [bw], delay, loss_rate, queue_size)
        return ret_trace

    # use bandwidth generator.
    assert T_l_range is not None and len(
        T_l_range) == 2 and T_l_range[0] <= T_l_range[1]
    assert T_s_range is not None and len(
        T_s_range) == 2 and T_s_range[0] <= T_s_range[1]
    assert cov_range is not None and len(
        cov_range) == 2 and cov_range[0] <= cov_range[1]
    assert steps_range is not None and len(
        steps_range) == 2 and steps_range[0] <= steps_range[1]
    assert timestep_range is not None and len(
        timestep_range) == 2 and timestep_range[0] <= timestep_range[1]
    T_l = float(np.random.uniform(T_l_range[0], T_l_range[1], 1))
    T_s = float(np.random.uniform(T_s_range[0], T_s_range[1], 1))
    cov = float(np.random.uniform(cov_range[0], cov_range[1], 1))
    steps = int(np.random.randint(steps_range[0], steps_range[1]+1, 1))
    timestep = float(np.random.uniform(
        timestep_range[0], timestep_range[1], 1))

    timestamps, bandwidths = generate_bw_series(
        T_l, T_s, cov, duration, steps, bandwidth_range[0],
        bandwidth_range[1], timestep)
    ret_trace = Trace(timestamps, bandwidths, delay, loss_rate, queue_size)
    return ret_trace


def generate_traces(config_file: str, tot_trace_cnt: int, duration: int,
                    constant_bw: bool = True):
    config = read_json_file(config_file)
    traces = []

    for env_config in config:
        bw_min, bw_max = env_config['bandwidth']
        delay_min, delay_max = env_config['delay']
        loss_min, loss_max = env_config['loss']
        queue_min, queue_max = env_config['queue']
        if 'duration' in env_config:
            duration_min, duration_max = env_config['duration']
        else:
            duration_min, duration_max = duration, duration

        # used by bandwidth generation
        T_l_min, T_l_max = env_config['T_l'] if 'T_l' in env_config else 50, 50
        T_s_min, T_s_max = env_config['T_s'] if 'T_s' in env_config else 10, 10
        cov_min, cov_max = env_config['cov'] if 'cov' in env_config else 0.2, 0.2
        steps_min, steps_max = env_config['steps'] if 'steps' in env_config else 10, 10
        timestep_min, timestep_max = env_config['timestep'] if 'timestep' in env_config else 1, 1
        trace_cnt = int(round(env_config['weight'] * tot_trace_cnt))
        for _ in range(trace_cnt):
            trace = generate_trace((duration_min, duration_max),
                                   (bw_min, bw_max),
                                   (delay_min, delay_max),
                                   (loss_min, loss_max),
                                   (queue_min, queue_max),
                                   (T_l_min, T_l_max),
                                   (T_s_min, T_s_max),
                                   (cov_min, cov_max),
                                   (steps_min, steps_max),
                                   (timestep_min, timestep_max),
                                   constant_bw=constant_bw)
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


def generate_bw_series(T_l: float, T_s: float, cov: float, duration: float,
                       steps: int, min_bw: float, max_bw: float,
                       timestep: float = 1):
    """Generate a time-variant bandwidth series.

    Args:
        T_l: prob of staying in one state = 1 - 1 / T_l. Value range: [0, inf)
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

    # probability to stay in same state
    prob_stay = 1 - 1 / T_l

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
