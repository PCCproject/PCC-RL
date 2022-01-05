import argparse
import copy
import os
import time

import numpy as np

from common.utils import read_json_file, set_seed
from simulator.network_simulator.bbr import BBR
from simulator.network_simulator.cubic import Cubic
from simulator.network_simulator.bbr_old import BBR_old
from simulator.aurora import test_on_traces
from simulator.trace import generate_trace


DEFAULT_VALUES = {"bandwidth_lower_bound": 0.1,
                  "bandwidth_upper_bound": 3.16,
                  "delay":  101,
                  "loss": 0.0,
                  "queue": 1.6,
                  "T_s": 7.5,
                  "duration": 30,
                  "delay_noise": 0}


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Aurora testing in simulator to prepare "
                                     "data for heat map.")
    parser.add_argument('--save-dir', type=str, default="",
                        help="direcotry to testing results.")
    parser.add_argument('--cc', type=str, required=True,
                        choices=('bbr', 'cubic', 'bbr_old', 'genet_bbr_old', 'genet_bbr',
                                 'genet_cubic', 'pretrained', 'overfit_config'),
                        help='Rule-based congestion control name.')
    parser.add_argument('--models-path', type=str,
                        help="path to genet trained Aurora models.")
    parser.add_argument('--config-file', type=str, required=True,
                        help="path to configuration file.")
    parser.add_argument('--dims', type=str, required=True, nargs=2,
                        help="2 dimenstions used to compare. Others use default values.")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--nproc', type=int, default=24, help='proc cnt')

    args, _ = parser.parse_known_args()
    return args

def get_dim_vals(dim: str):
    if dim == 'bandwidth_upper_bound':
        dim_vals = [0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0]
    elif dim == 'delay':
        dim_vals = [2, 5, 8, 10, 20, 50, 80, 100, 120, 150, 180, 200]
    elif dim == 'loss':
        dim_vals = [0, 0.0001, 0.0002, 0.0005, 0.0008, 0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.05]
    elif dim == 'queue':
        dim_vals = [0.1, 0.2, 0.5, 0.8, 1, 1.2, 1.6, 1.8, 2, 2.2, 2.5, 2.8, 3]
    elif dim == 'T_s':
        dim_vals = [0.1, 0.4, 0.6, 0.8, 1.2, 1.6, 2.0, 5, 8, 10.0, 12.0, 15.0, 17.0, 25, 30]
        # dim_vals = [1, 2, 3, 4, 5, 6, 7, 8, 10.0, 12.0, 15.0, 17.0, 20, 25, 28, 30]
    else:
        raise NotImplementedError
    return dim_vals


def main():
    args = parse_args()
    set_seed(args.seed)
    # tokens = os.path.basename(os.path.dirname(os.path.dirname(args.save_dir))).split('_')
    # config0_dim0_idx = int(tokens[1])
    # config0_dim1_idx = int(tokens[2])
    # config1_dim0_idx = int(tokens[4])
    # config1_dim1_idx = int(tokens[5])

    dim0, dim1 = args.dims
    config = read_json_file(args.config_file)[0]
    assert dim0 in config and dim1 in config

    # dim0_vals = np.linspace(config[dim0][0], config[dim0][1], 10)
    # dim1_vals = np.linspace(config[dim1][0], config[dim1][1], 10)
    dim0_vals = get_dim_vals(dim0)
    dim1_vals = get_dim_vals(dim1)
    print(dim0_vals)
    print(dim1_vals)
    traces = []
    save_dirs = []
    with open('heatmap_trace_cnt_ratio.npy', 'rb') as f:
        cnt_ratio = np.load(f)
    for dim0_idx, dim0_val in enumerate(dim0_vals):
        for dim1_idx, dim1_val in enumerate(dim1_vals):
            dim_vals = copy.copy(DEFAULT_VALUES)
            dim_vals[dim0] = dim0_val
            dim_vals[dim1] = dim1_val
            # print(i, dim0_val, dim1_val, dim_vals)
            cnt = 10
            # if cnt_ratio[dim0_idx, dim1_idx] > 1:
            #     cnt *= int(cnt_ratio[dim0_idx, dim1_idx])
            # print(cnt)
            for trace_idx in range(cnt):
                trace = generate_trace(
                    duration_range=(dim_vals['duration'], dim_vals['duration']),
                    bandwidth_lower_bound_range=(dim_vals['bandwidth_lower_bound'],
                                                 dim_vals['bandwidth_lower_bound']),
                    bandwidth_upper_bound_range=(dim_vals['bandwidth_upper_bound'],
                                                 dim_vals['bandwidth_upper_bound']),
                    delay_range=(dim_vals['delay'], dim_vals['delay']),
                    loss_rate_range=(dim_vals['loss'], dim_vals['loss']),
                    queue_size_range=(dim_vals['queue'], dim_vals['queue']),
                    T_s_range=(dim_vals['T_s'], dim_vals['T_s']),
                    delay_noise_range=(dim_vals['delay_noise'],
                                       dim_vals['delay_noise']))
                traces.append(trace)
                save_dir = os.path.join(args.save_dir, 'pair_{}_{}'.format(dim0_idx, dim1_idx), 'trace_{}'.format(trace_idx))
                save_dirs.append(save_dir)
                os.makedirs(save_dir, exist_ok=True)
                trace.dump(os.path.join(save_dir, 'trace_{}.json'.format(trace_idx)))
    if args.cc == 'genet_bbr' or args.cc == 'genet_cubic' or args.cc == 'genet_bbr_old':
        genet_seed = ''
        for s in args.models_path.split('/'):
            if 'seed' in s:
                genet_seed = s
        for bo in range(0, 30, 3):
        # for bo_dir in natural_sort(glob.glob(os.path.join(args.models_path, "bo_*/"))):
            bo_dir = os.path.join(args.models_path, "bo_{}".format(bo))
            step = 64800
            model_path = os.path.join(bo_dir, 'model_step_{}.ckpt'.format(step))
            if not os.path.exists(model_path + '.meta'):
                print(model_path, 'does not exist')
                continue
            print(model_path)
            genet_save_dirs = [os.path.join(
                save_dir, args.cc, genet_seed, "bo_{}".format(bo),
                "step_{}".format(step)) for save_dir in save_dirs]
            t_start = time.time()
            test_on_traces(model_path, traces, genet_save_dirs, args.nproc, 42, False, False)
            print('bo {}: {:.3f}'.format(bo, time.time() - t_start))
    elif args.cc == 'pretrained':
        pretrained_save_dirs = [os.path.join(save_dir, args.cc) for save_dir in save_dirs]
        t_start = time.time()
        test_on_traces(args.models_path, traces, pretrained_save_dirs, args.nproc, 42, False, False)
        print('pretrained: {:.3f}'.format(time.time() - t_start))
    elif args.cc == 'overfit_config':
        overfit_config_save_dirs = [os.path.join(save_dir, args.cc) for save_dir in save_dirs]
        t_start = time.time()
        test_on_traces(args.models_path, traces, overfit_config_save_dirs, args.nproc, 42, False, False)
        print('overfit_config: {:.3f}'.format(time.time() - t_start))
    else:
        if args.cc == 'bbr':
            cc = BBR(False)
        elif args.cc == 'cubic':
            cc = Cubic(False)
        elif args.cc == 'bbr_old':
            cc = BBR_old(False)
        else:
            raise NotImplementedError
        heuristic_save_dirs = [os.path.join(save_dir, cc.cc_name) for save_dir in save_dirs]
        t_start = time.time()
        cc.test_on_traces(traces, heuristic_save_dirs, False, args.nproc)
        print('{}: {:.3f}'.format(args.cc, time.time() - t_start))


if __name__ == "__main__":
    main()
