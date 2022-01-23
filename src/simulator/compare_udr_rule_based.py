import argparse
import os

from common.utils import set_seed
from simulator.aurora import test_on_traces
from simulator.network_simulator.bbr_old import BBR_old
from simulator.network_simulator.cubic import Cubic
from simulator.trace import generate_traces


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("BO training in simulator.")
    parser.add_argument('--save-dir', type=str, required=True, help="directory"
                        " to save testing and intermediate results.")
    parser.add_argument('--model-path', type=str, default=None,
                        help="path to Aurora model to start from.")
    parser.add_argument("--config-file", type=str, required=True,
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
    else:
        raise NotImplementedError

    traces = generate_traces(args.config_file, 20, 30)
    save_dirs = [os.path.join(args.save_dir, "trace_{:02d}".format(i))
                 for i in range(len(traces))]
    cc_save_dirs = [os.path.join(save_dir, cc.cc_name)
                    for save_dir in save_dirs]
    cc.test_on_traces(traces, cc_save_dirs, plot_flag=True, n_proc=8)

    aurora_save_dirs = [os.path.join(save_dir, 'aurora')
                        for save_dir in save_dirs]
    test_on_traces(args.model_path, traces, aurora_save_dirs, nproc=8, seed=42,
                   record_pkt_log=False, plot_flag=True)

    for i, (trace, save_dir) in enumerate(zip(traces, save_dirs)):
        trace.dump(os.path.join(save_dir, 'trace_{:02d}.json'.format(i)))


if __name__ == '__main__':
    main()
