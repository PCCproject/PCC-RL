import glob
import os
from typing import Union

from simulator.trace import Trace, generate_traces
from common.utils import set_seed


class SyntheticDataset:
    def __init__(self, count: int, config_file: Union[str, None],
                 seed: int = 42):
        set_seed(seed)
        self.count = count
        self.traces = []
        self.config_file = config_file
        if self.config_file:
            self.traces = generate_traces(self.config_file, self.count, 30)
        assert len(self.traces) == self.count

    def dump(self, save_dir: str):
        """Save all traces."""
        os.makedirs(save_dir, exist_ok=True)
        for i, trace in enumerate(self.traces):
            trace.dump(os.path.join(save_dir, 'trace_{:05d}.json'.format(i)))

    @staticmethod
    def load_from_file(trace_file: str):
        traces = []
        with open(trace_file, 'r') as f:
            for line in f:
                line = line.strip()
                traces.append(Trace.load_from_file(line))
        dataset = SyntheticDataset(len(traces), None)
        dataset.traces = traces
        return dataset

    @staticmethod
    def load_from_dir(trace_dir: str):
        files = sorted(glob.glob(os.path.join(trace_dir, 'trace_*.json')))
        traces = []
        for file in files:
            traces.append(Trace.load_from_file(file))
        dataset = SyntheticDataset(len(traces), None)
        dataset.traces = traces
        return dataset

    def __len__(self):
        assert len(self.traces) == self.count
        return self.count

    def __getitem__(self, idx):
        return self.traces[idx]

dataset = SyntheticDataset(1000, '../../config/train/udr_7_dims_0826/udr_large.json')
dataset.dump('/datamirror/zxxia/PCC-RL/results_1006/synthetic_dataset')
