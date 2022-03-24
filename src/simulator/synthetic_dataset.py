import glob
import os
from typing import Optional

import numpy as np

from simulator.trace import Trace, generate_traces, generate_traces_from_config
from common.utils import set_seed, read_json_file, zero_one_normalize


class SyntheticDataset:
    def __init__(self, count: int, config_file: Optional[str], config=None,
                 seed: int = 42):
        set_seed(seed)
        self.count = count
        self.traces = []
        self.config_file = config_file
        self.config = config
        if self.config_file:
            self.traces = generate_traces(self.config_file, self.count, 30)
        elif self.config:
            self.traces = generate_traces_from_config(self.config, self.count, 30)

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

    def prepare_data_for_DoppelGANger(self, save_dir: str = ""):
        data_feature = []
        data_attribute = []
        for trace in self.traces:
            sample_feature = np.array(trace.bandwidths).reshape(-1, 1)
            data_feature.append(sample_feature)
            data_attribute.append(np.array([1, 0]))

        print(np.min(data_feature))
        print(np.max(data_feature))

        data_feature = zero_one_normalize(np.stack(data_feature))
        data_attribute = np.stack(data_attribute)

        data_gen_flag = np.ones(data_feature[:,:, 0].shape)

        np.savez(os.path.join(save_dir, 'data_train.npz'),
                 data_feature=data_feature, data_attribute=data_attribute,
                 data_gen_flag=data_gen_flag)
        np.savez(os.path.join(save_dir, 'data_train_no_attribute.npz'),
                 data_feature=data_feature, data_attribute=data_attribute,
                 data_gen_flag=data_gen_flag)

# dataset = SyntheticDataset(1000, '../../config/train/udr_7_dims_0826/udr_large.json')
# dataset.dump('/datamirror/zxxia/PCC-RL/results_1006/synthetic_dataset')
# config = read_json_file('../../config/train/udr_7_dims_0826/udr_large.json')
#
# dataset = SyntheticDataset(1000, '../../config/train/udr_7_dims_0826/udr_large.json')
# dataset.dump('/datamirror/zxxia/PCC-RL/results_1006/synthetic_dataset')

def main():
    # dataset = SyntheticDataset(1000, '../../config/train/udr_7_dims_0826/udr_large.json')
    # dataset.dump('/datamirror/zxxia/PCC-RL/results_1006/synthetic_dataset')
    # config = read_json_file('../../config/train/udr_7_dims_0826/udr_large.json')
    #
    # dataset = SyntheticDataset(1000, '../../config/train/udr_7_dims_0826/udr_large.json')
    # dataset.dump('/datamirror/zxxia/PCC-RL/results_1006/synthetic_dataset')

    dataset = SyntheticDataset(100, '../../config/train/aurora.json')
    dataset.prepare_data_for_DoppelGANger('../../../DoppelGANger/data/constant_bw')

if __name__ == '__main__':
    main()
