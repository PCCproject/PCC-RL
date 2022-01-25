"""Evaluate all training models and rule-based methods on a set of real traces.
Used to generate Figure 12.
"""
import argparse
import os

import pandas as pd

import numpy as np

from simulator.aurora import test_on_traces
from simulator.network_simulator.bbr import BBR
from simulator.network_simulator.bbr_old import BBR_old
from simulator.network_simulator.cubic import Cubic
# from simulator.network_simulator.pcc.vivace.vivace_latency import VivaceLatency
# from simulator.pantheon_dataset import PantheonDataset
from simulator.synthetic_dataset import SyntheticDataset

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

TARGET_CCS = ["bbr", "cubic", "vegas", "indigo", "ledbat", "quic"]


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Aurora Testing in simulator.")
    parser.add_argument('--save-dir', type=str, default="",
                        help="direcotry to testing results.")
    parser.add_argument('--cc', type=str, required=True,
                        # choices=("bbr", 'bbr_old', "cubic", "udr1", "udr2", "udr3",
                        #          "genet_bbr", 'genet_bbr_old', 'genet_cubic',
                        #          'real', 'cl1', 'cl2', 'pretrained', 'cl2_new', 'real_cellular', ),
                        help='congestion control name')
    parser.add_argument('--models-path', type=str, default="",
                        help="path to Aurora models.")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--nproc', type=int, default=16, help='proc cnt')

    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    # dataset = SyntheticDataset(25, '/tank/zxxia/PCC-RL/config/train/udr_7_dims_0826/udr_large.json', seed=42)
    dataset = SyntheticDataset.load_from_dir('/datamirror/zxxia/PCC-RL/results_1006/synthetic_dataset')
    traces = dataset.traces
    # cnt = 0
    # for idx, trace in enumerate(traces):
    #     if trace.min_bw > 0.5:
    #         cnt+=1
    #         print(idx)
    # print(cnt)
    # return
    save_dirs = [os.path.join(args.save_dir, 'trace_{:05d}'.format(i)) for i in range(len(dataset))]

    if args.cc == 'bbr':
        cc = BBR(False)
        cc.test_on_traces(traces, [os.path.join(save_dir, cc.cc_name)
                                   for save_dir in save_dirs], True, args.nproc)
    elif args.cc == 'bbr_old':
        cc = BBR_old(False)
        cc.test_on_traces(traces, [os.path.join(save_dir, cc.cc_name)
                                   for save_dir in save_dirs], True, args.nproc)
    elif args.cc == 'cubic':
        cc = Cubic(False)
        cc.test_on_traces(traces, [os.path.join(save_dir, cc.cc_name)
                                   for save_dir in save_dirs], True, args.nproc)
    elif args.cc == 'real':
        model_path = args.models_path
        real_save_dirs = [os.path.join(
            save_dir, args.cc, "seed_{}".format(args.seed)) for save_dir in save_dirs]
        test_on_traces(model_path, traces, real_save_dirs,
                       args.nproc, 42, False, plot_flag=True)
    elif args.cc == 'pretrained':
        udr_seed = ''
        for s in args.models_path.split('/'):
            if 'seed' in s:
                udr_seed = s
        step = 7200
        while step <= 151200:
            if not os.path.exists(os.path.join(args.models_path, 'model_step_{}.ckpt.meta'.format(step))):
                break
            udr_save_dirs = [os.path.join(
                save_dir, args.cc, udr_seed, "step_{}".format(step)) for save_dir in save_dirs]
            model_path = os.path.join(
                args.models_path, 'model_step_{}.ckpt'.format(step))
            test_on_traces(model_path, traces, udr_save_dirs,
                           args.nproc, 42, False, True)
            step += 28800
    elif args.cc == 'udr1' or args.cc == 'udr2' or args.cc == 'udr3' or \
            args.cc == 'cl1' or args.cc == 'cl1_new' or args.cc == 'cl2' or args.cc == 'cl2_new' or args.cc == 'real_cellular' or 'udr' in args.cc or 'cl' in args.cc:
        # TODO: bug here when there is no validation log
        # original implementation
        # val_log = pd.read_csv(os.path.join(
        #     args.models_path, 'validation_log.csv'), sep='\t')
        # for idx in np.linspace(0, len(val_log['num_timesteps']) / 2 - 1, 10):
        #     step = int(val_log['num_timesteps'].iloc[int(idx)])
        #     udr_seed = ''
        #     for s in args.models_path.split('/'):
        #         if 'seed' in s:
        #             udr_seed = s
        #     udr_save_dirs = [os.path.join(
        #         save_dir, args.cc, udr_seed, "step_{}".format(step)) for save_dir in save_dirs]
        #     model_path = os.path.join(
        #         args.models_path, 'model_step_{}.ckpt'.format(step))
        #     test_on_traces(model_path, traces, udr_save_dirs,
        #                    args.nproc, 42, False, False)
        # new implementation to be fair with genet
        udr_seed = ''
        for s in args.models_path.split('/'):
            if 'seed' in s:
                udr_seed = s
        step = 0
        while step <= 720000:
            if not os.path.exists(os.path.join(args.models_path, 'model_step_{}.ckpt.meta'.format(step))):
                break
            udr_save_dirs = [os.path.join(
                save_dir, args.cc, udr_seed, "step_{}".format(step)) for save_dir in save_dirs]
            model_path = os.path.join(
                args.models_path, 'model_step_{}.ckpt'.format(step))
            test_on_traces(model_path, traces, udr_save_dirs,
                           args.nproc, 42, False, True)
            # step += (7200) * 2
            step += 72000
            print(step)
    # elif args.cc == 'genet_bbr' or args.cc == 'genet_cubic' or 'genet_bbr_old':
    elif 'genet' in args.cc: #== 'genet_bbr' or args.cc == 'genet_cubic' or 'genet_bbr_old': genet_seed = ''
        for s in args.models_path.split('/'):
            if 'seed' in s:
                genet_seed = s
        # for bo in range(0, 15, 3):
        for bo in range(0, 10):
            bo_dir = os.path.join(args.models_path, "bo_{}".format(bo))
            for step in range(64800, 72000, 14400):
            # step = 64800
                model_path = os.path.join(
                    bo_dir, 'model_step_{}.ckpt'.format(step))
                if not os.path.exists(model_path + '.meta'):
                    print("skip " + model_path + '.meta')
                    continue
                genet_save_dirs = [os.path.join(
                    save_dir, args.cc, genet_seed, "bo_{}".format(bo),
                    "step_{}".format(step)) for save_dir in save_dirs]
                # print(genet_save_dirs)
                test_on_traces(model_path, traces, genet_save_dirs,
                               args.nproc, 42, False, True)
    else:
        raise ValueError


if __name__ == "__main__":
    main()
