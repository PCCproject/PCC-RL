"""Evaluate all training models and rule-based methods on a set of real traces.
Used to generate Figure 12.
"""
import argparse
import glob
import sys
import os
import multiprocessing as mp
import time

from tqdm import tqdm
import pandas as pd

import numpy as np

from common.utils import natural_sort
from simulator.aurora import Aurora
from simulator.network_simulator.bbr import BBR
from simulator.network_simulator.cubic import Cubic
# from simulator.network_simulator.pcc.vivace.vivace_latency import VivaceLatency
from simulator.trace import Trace

# UDR1_ROOT = '/Users/zxxia/Project/PCC-RL/models/udr_small_lossless/seed_20'
# UDR2_ROOT = '/Users/zxxia/Project/PCC-RL/models/udr_mid_lossless/seed_20'
# UDR3_ROOT = '/Users/zxxia/Project/PCC-RL/models/udr_large_lossless/seed_20'
# UDR1_ROOT = '../results/resmodels/udr_2/udr_small_seed_20/seed_20'
UDR_ROOT = '../../results_0826/udr_6'
UDR_ROOT = '../../results_0826/udr_7'
UDR3_ROOT = '/Users/zxxia/Project/PCC-RL/models/udr_2/udr_large/seed_20'
# GENET_MODEL_PATH = "test/bo_3/model_step_72000.ckpt"
GENET_MODEL_PATH = "../../models/udr_large_lossless/seed_20/model_step_2124000.ckpt"
GENET_MODEL_PATH = "../../models/udr_large_lossless/seed_20/model_step_2124000.ckpt"
# GENET_MODEL_PATH = "test/bo_3/model_step_72000.ckpt"
# GENET_MODEL_PATH = "../../models/bo_10_model_step_36000/bo_10_model_step_36000.ckpt"


RESULT_ROOT = "../../results_0826"
RESULT_ROOT = "../../results_0910"
TRACE_ROOT = "../../data"
# TRACE_ROOT = "../../data/cellular/2018-12-11T00-27-AWS-Brazil-2-to-Colombia-cellular-3-runs-3-flows"

TARGET_CCS = ["bbr", "cubic", "vegas", "indigo", "ledbat", "quic"]
TARGET_CCS = ["bbr"] #, "cubic", "vegas", "indigo", "ledbat", "quic"]


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Aurora Testing in simulator.")
    parser.add_argument('--save-dir', type=str, default="",
                        help="direcotry to testing results.")
    parser.add_argument('--cc', type=str, required=True,
                        choices=("bbr", "cubic", "udr1", "udr2", "udr3",
                                 "genet_bbr", 'genet_cubic'),
                        help='congestion control name')
    parser.add_argument("--conn-type", type=str,
                        choices=('ethernet', 'cellular', 'wifi'),
                        help='connection type')
    parser.add_argument('--models-path', type=str, default="",
                        help="path to Aurora models.")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--nproc', type=int, default=4, help='proc cnt')

    args, _ = parser.parse_known_args()
    return args

def test_cc_on_traces(cc, traces, save_dirs, nproc):
    print("Testing {} on real traces...".format(cc.cc_name))
    arguments = []
    for trace, save_dir in tqdm(zip(traces, save_dirs), total=len(traces)):
        os.makedirs(save_dir, exist_ok=True)
        arguments.append((trace, save_dir, True))
    n_proc = nproc
    with mp.Pool(processes=n_proc) as pool:
        pool.starmap(cc.test, arguments)

def test_aurora_on_trace(model_path, trace, save_dir, plot_flag):
    aurora = Aurora(seed=10, log_dir="", pretrained_model_path=model_path,
                    timesteps_per_actorbatch=10)
    ret = aurora.test(trace, save_dir, plot_flag)
    return ret

def main():
    args = parse_args()
    if args.conn_type == 'ethernet':
        queue_size = 500
    elif args.conn_type == 'cellular':
        queue_size = 50
    else:
        raise ValueError
    link_dirs = glob.glob(os.path.join(TRACE_ROOT, args.conn_type, "*/"))
    for link_dir in link_dirs:
        traces = []
        save_dirs = []
        link_name = link_dir.split('/')[-2]
        for cc in TARGET_CCS:
            print("Loading real traces collected by {}...".format(cc))
            for trace_file in tqdm(sorted(glob.glob(os.path.join(link_dir,"{}_datalink_run1.log".format(cc))))):
                traces.append(Trace.load_from_pantheon_file(trace_file, 0.0, queue_size, front_offset=5))
                save_dir = os.path.join(args.save_dir, args.conn_type, link_name, os.path.splitext(os.path.basename(trace_file))[0])
                os.makedirs(save_dir, exist_ok=True)
                save_dirs.append(save_dir)
        t_start = time.time()
        if args.cc == 'bbr':
            cc = BBR(True)
            cc.test_on_traces(traces, [os.path.join(save_dir, cc.cc_name)
                                       for save_dir in save_dirs], True, args.nproc)
        elif args.cc == 'cubic':
            cc = Cubic(True)
            cc.test_on_traces(traces, [os.path.join(save_dir, cc.cc_name)
                                       for save_dir in save_dirs], True, args.nproc)
        elif args.cc == 'udr1' or args.cc == 'udr2' or args.cc == 'udr3':
            val_log = pd.read_csv(os.path.join(args.models_path, 'validation_log.csv'), sep='\t')
            for idx in np.linspace(0, len(val_log['num_timesteps']) / 2 - 1, 10):
            # np.concatenate(
            #     [np.array([0]),
            #      np.exp(np.linspace(np.log(1), np.log(len(val_log['num_timesteps']) - 1), 10))]):
                step = int(val_log['num_timesteps'].iloc[int(idx)])
                udr_seed = ''
                for s in args.models_path.split('/'):
                    if 'seed' in s:
                        udr_seed = s
                udr_save_dirs = [os.path.join(save_dir, args.cc, udr_seed, "step_{}".format(step)) for save_dir in save_dirs]
                model_path = os.path.join(args.models_path, 'model_step_{}.ckpt'.format(step))
                arguments = []
                for trace, save_dir in zip(traces, udr_save_dirs):
                    arguments.append((model_path, trace, save_dir, True))
                with mp.Pool(processes=args.nproc) as pool:
                    pool.starmap(test_aurora_on_trace, arguments)
        elif args.cc == 'genet_bbr' or args.cc == 'genet_cubic':
            for bo in range(20):
            # for bo_dir in natural_sort(glob.glob(os.path.join(args.models_path, "bo_*/"))):
                bo_dir = os.path.join(args.models_path, "bo_{}".format(bo))
                step = 64800
                model_path = os.path.join(bo_dir, 'model_step_{}.ckpt'.format(step))
                if not os.path.exists(model_path + '.meta'):
                    continue
                genet_save_dirs = [os.path.join(save_dir, args.cc, "bo_{}".format(bo), "step_{}".format(step)) for save_dir in save_dirs]
                arguments = []
                for trace, save_dir in zip(traces, genet_save_dirs):
                    arguments.append((model_path, trace, save_dir, True))
                with mp.Pool(processes=args.nproc) as pool:
                    pool.starmap(test_aurora_on_trace, arguments)
        else:
            raise ValueError

if __name__ == "__main__":
    main()
