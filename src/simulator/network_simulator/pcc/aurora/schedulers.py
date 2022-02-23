from typing import List

import numpy as np

from simulator.network_simulator.bbr import BBR
from simulator.network_simulator.bbr_old import BBR_old
from simulator.network_simulator.cubic import Cubic
from simulator.trace import Trace, generate_traces


class Scheduler:
    def get_trace(self):
        raise NotImplemented


class TestScheduler(Scheduler):
    def __init__(self, trace: Trace):
        self.trace = trace

    def get_trace(self):
        return self.trace


class UDRTrainScheduler(Scheduler):
    def __init__(
        self, config_file: str, traces: List[Trace], percent: float = 0.0
    ):
        self.config_file = config_file
        self.traces = traces
        self.percent = percent

    def get_trace(self):
        if self.traces and np.random.uniform(0, 1) < self.percent:
            return np.random.choice(self.traces)
        elif self.config_file:
            return generate_traces(self.config_file, 1, duration=30)[0]
        else:
            raise ValueError


class CL1TrainScheduler(Scheduler):
    def __init__(self, config_files: List[str], aurora):
        assert config_files
        self.config_files = config_files
        self.config_file = self.config_files[0]
        self.aurora = aurora

    def get_trace(self):
        if self.aurora.callback.n_calls % self.aurora.callback.check_freq == 0:
            i = int(
                self.aurora.callback.n_calls
                // (self.aurora.callback.check_freq * 20)
            )
            self.config_file = self.config_files[i]
            # print(self.n_calls, i, self.model.env.config_file)
        return generate_traces(self.config_file, 1, duration=30)[0]


class CL2TrainScheduler(Scheduler):
    def __init__(self, config_file: str, aurora, baseline: str):
        self.config_file = config_file
        self.aurora = aurora
        self.difficulty_trace_cache = {0: [], 1: [], 2: [], 3: [], 4: []}
        if baseline == "bbr":
            self.baseline = BBR()
        elif baseline == "bbr_old":
            self.baseline = BBR_old()
        elif baseline == "cubic":
            self.baseline = Cubic()
        else:
            raise ValueError
        self.difficulty_level = 0

    def get_trace(self):
        if self.aurora.callback.n_calls % self.aurora.callback.check_freq == 0:
            self.difficulty_level = int(
                self.aurora.callback.n_calls
                // (self.aurora.callback.check_freq * 20)
            )
        return self._sample_trace()

    def _sample_trace(self):
        if self.difficulty_level == 0:
            target_difficulty = self.difficulty_level
        elif self.difficulty_level == 1:
            prob = np.random.uniform(0, 1, 1).item()
            if prob < 0.7:
                target_difficulty = 0
            else:
                target_difficulty = 1
        elif self.difficulty_level == 2:
            prob = np.random.uniform(0, 1, 1).item()
            if prob < 0.49:
                target_difficulty = 0
            elif 0.49 < prob < 0.7:
                target_difficulty = 1
            else:
                target_difficulty = 2
        elif self.difficulty_level == 3:
            prob = np.random.uniform(0, 1, 1).item()
            if prob < 0.343:
                target_difficulty = 0
            elif 0.343 < prob < 0.49:
                target_difficulty = 1
            elif 0.49 < prob < 0.7:
                target_difficulty = 2
            else:
                target_difficulty = 3
        else:
            prob = np.random.uniform(0, 1, 1).item()
            if prob < 0.2401:
                target_difficulty = 0
            elif 0.2401 < prob < 0.343:
                target_difficulty = 1
            elif 0.343 < prob < 0.49:
                target_difficulty = 2
            elif 0.49 < prob < 0.7:
                target_difficulty = 3
            else:
                target_difficulty = 4
        while not self.difficulty_trace_cache[target_difficulty]:
            traces = generate_traces(self.config_file, 1, duration=30)
            self._insert_to_difficulty_cache(traces)

        trace = np.random.choice(self.difficulty_trace_cache[target_difficulty])
        self.difficulty_trace_cache[target_difficulty].remove(trace)
        print("select from level", target_difficulty)
        return trace

    def _insert_to_difficulty_cache(self, traces):
        len_cache = 100
        for _, trace in enumerate(traces):
            difficulty = trace.optimal_reward - self.baseline.test(trace, "")[1]
            # if difficulty < 137.6:
            #     diff_key = 0
            # elif 137.6 <= difficulty < 233.4:
            #     diff_key = 1
            # elif 233.4 <= difficulty < 344.2:
            #     diff_key = 2
            # elif 344.2 <= difficulty < 455.1:
            #     diff_key = 3
            # else:
            #     diff_key = 4
            # if trace.avg_bw < 1:
            #     diff_key = 0
            if difficulty < 210.5:
                diff_key = 0
            elif 210.5 <= difficulty < 314.7:
                diff_key = 1
            elif 314.7 <= difficulty < 410:
                diff_key = 2
            elif 410 <= difficulty < 498:
                diff_key = 3
            else:
                diff_key = 4

            if len(self.difficulty_trace_cache[diff_key]) >= len_cache:
                self.difficulty_trace_cache[diff_key].pop(0)
            self.difficulty_trace_cache[diff_key].append(trace)
