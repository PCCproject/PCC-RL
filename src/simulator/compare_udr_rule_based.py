import argparse
import os

import numpy as np

from common.utils import set_seed, read_json_file
from simulator.aurora import test_on_traces
from simulator.network_simulator.bbr_old import BBR_old
from simulator.network_simulator.bbr import BBR
from simulator.network_simulator.cubic import Cubic
from simulator.trace import generate_traces, generate_traces_from_config
from simulator.pantheon_dataset import PantheonDataset


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("BO training in simulator.")
    parser.add_argument('--save-dir', type=str, required=True, help="directory"
                        " to save testing and intermediate results.")
    parser.add_argument('--model-path', type=str, default=None,
                        help="path to Aurora model to start from.")
    parser.add_argument("--config-file", type=str, default="",
                        help="Path to configuration file.")
    parser.add_argument("--seed", type=int, required=True, help="seed")
    parser.add_argument('--heuristic', type=str, default="cubic",
                        choices=('bbr', 'bbr_old', 'cubic', 'optimal'),
                        help='Congestion control rule based method.')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.heuristic == 'cubic':
        cc = Cubic()
    elif args.heuristic == 'bbr_old':
        cc = BBR_old()
    elif args.heuristic == 'bbr':
        cc = BBR()
    else:
        raise NotImplementedError
    # if 'large' in args.config_file:
    #     config = read_json_file(args.config_file)
    #     config[0]['bandwidth_lower_bound'] = (1, 1)
    #     config[0]['bandwidth_upper_bound'] = (1, 100)
    #     traces = generate_traces_from_config(config, 50, 30)
    # else:
    if not args.config_file:
        dataset = PantheonDataset('../../data', 'all')
        traces = dataset.get_traces(0, 50)
        save_dirs = [os.path.join(
            args.save_dir, link_conn_type, link_name, trace_name)
            for link_conn_type, (link_name, trace_name) in
            zip(dataset.link_conn_types, dataset.trace_names)]
    else:
        traces = generate_traces(args.config_file, 50, 30)
        save_dirs = [os.path.join(args.save_dir, "trace_{:02d}".format(i))
                     for i in range(len(traces))]
    cc_save_dirs = [os.path.join(save_dir, cc.cc_name)
                    for save_dir in save_dirs]
    cc_res = cc.test_on_traces(traces, cc_save_dirs, plot_flag=False, n_proc=16)

    aurora_save_dirs = [os.path.join(save_dir, 'aurora')
                        for save_dir in save_dirs]
    aurora_res = test_on_traces(args.model_path, traces, aurora_save_dirs,
                                nproc=16, seed=42, record_pkt_log=False,
                                plot_flag=False)
    print(cc.cc_name, np.mean([res[1] for res in cc_res]))
    print('aurora', np.mean([res[1] for res in aurora_res]))

    for i, (trace, save_dir) in enumerate(zip(traces, save_dirs)):
        trace.dump(os.path.join(save_dir, 'trace_{:02d}.json'.format(i)))


if __name__ == '__main__':
    main()
