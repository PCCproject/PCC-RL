"""Evaluate all training models and rule-based methods on a set of real traces.
Used to generate Figure 12.
"""
import glob
import os
import multiprocessing as mp

from tqdm import tqdm

import numpy as np

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
# GENET_MODEL_PATH = "test/bo_3/model_step_72000.ckpt"
# GENET_MODEL_PATH = "../../models/bo_10_model_step_36000/bo_10_model_step_36000.ckpt"

GENET_ROOT = "../../results_0826/genet_cubic_exp_2"
GENET_ROOT = "../../results_0826/genet_bbr_exp_1"

RESULT_ROOT = "../../results_0826"
TRACE_ROOT = "../../data/cellular/2019-09-17T22-29-AWS-California-1-to-Stanford-cellular-3-runs"
# TRACE_ROOT = "../../data/cellular/2018-12-11T00-27-AWS-Brazil-2-to-Colombia-cellular-3-runs-3-flows"
EXP_NAME = "train_perf2"
EXP_NAME = "train_perf2_noisy"

TARGET_CCS = ["bbr", "cubic", "vegas", "indigo", "ledbat", "quic"]
TARGET_CCS = ["bbr"] #, "cubic", "vegas", "indigo", "ledbat", "quic"]


def run_genet_model(root, traces, save_dirs, seed):
    for bo in range(9):
        for step in [14400, 50400,7200,  72000, 21600, 28800]:
            step = int(step)
            print('step', step)
            model_path=os.path.join(
                root, "bo_{}".format(bo), 'model_step_{}.ckpt'.format(step))
            if not os.path.exists(model_path + ".meta"):
                print(model_path)
                continue
            genet = Aurora(seed=seed, log_dir="", pretrained_model_path=model_path,
                timesteps_per_actorbatch=10)
            # test_cc_on_traces(genet, traces, [os.path.join(
            #     save_dir, "genet_cubic", "bo_{}".format(bo),
            #     "step_{}".format(step)) for save_dir in save_dirs])
            test_cc_on_traces(genet, traces, [os.path.join(
                save_dir, "genet_bbr", "bo_{}".format(bo),
                "step_{}".format(step)) for save_dir in save_dirs])

def run_udr_model(root, traces, save_dirs, udr_name, seed):
    # print(len(np.arange(7200, 2e5, 7200 * 2)))
    for step in np.arange(7200, 4e5, 7200 * 2):
        step = int(step)
        print('step', step)
        if not os.path.exists(os.path.join(
                root, 'model_step_{}.ckpt.meta'.format(step))):
            continue
        udr = Aurora(seed=seed, log_dir="", pretrained_model_path=os.path.join(
            root, 'model_step_{}.ckpt'.format(step)),
            timesteps_per_actorbatch=10)
        test_cc_on_traces(udr, traces, [os.path.join(
            save_dir, udr_name, 'seed_{}'.format(seed),
            "step_{}".format(step)) for save_dir in save_dirs])

def test_cc_on_traces(cc, traces, save_dirs):
    print("Testing {} on real traces...".format(cc.cc_name))
    for trace, save_dir in tqdm(zip(traces, save_dirs), total=len(traces)):
        os.makedirs(save_dir, exist_ok=True)
        cc.test(trace, save_dir, plot_flag=True)

def main():
    traces = []
    save_dirs = []
    for cc in TARGET_CCS:
        print("Loading real traces collected by {}...".format(cc))
        for trace_file in tqdm(sorted(glob.glob(os.path.join(
                TRACE_ROOT, "{}_datalink_run[1-3].log".format(cc))))):
            traces.append(Trace.load_from_pantheon_file(trace_file, 0.0, 50))
            save_dirs.append(os.path.join(
                RESULT_ROOT, EXP_NAME, os.path.basename(TRACE_ROOT),
                os.path.splitext(os.path.basename(trace_file))[0]))

    # bbr = BBR(True)
    # test_cc_on_traces(
    #     bbr, traces, [os.path.join(save_dir, bbr.cc_name) for save_dir in save_dirs])
    # cubic = Cubic(True)
    # test_cc_on_traces(
    #     cubic, traces, [os.path.join(save_dir, cubic.cc_name) for save_dir in save_dirs])


    cnt = 0
    # for step in np.arange(7200, 2e5, 14400):
    # for step in np.arange(12600, 43200, 1800):
    # for step in np.arange(12600, 43200, 1800):
    #     step = int(step)
    #     print('step', step)
    #     udr1 = Aurora(seed=20, log_dir="", pretrained_model_path=os.path.join(
    #                   UDR1_ROOT, 'model_step_{}.ckpt'.format(step)),
    #                   timesteps_per_actorbatch=10)
    #     udr2 = Aurora(seed=20, log_dir="",
    #                   pretrained_model_path=os.path.join(
    #                       UDR2_ROOT, 'model_step_{}.ckpt'.format(step)),
    #                   timesteps_per_actorbatch=10)
    #     udr3 = Aurora(seed=20, log_dir="",
    #                   pretrained_model_path=os.path.join(
    #                       UDR3_ROOT, 'model_step_{}.ckpt'.format(step)),
    #                   timesteps_per_actorbatch=10)
    #     # test_cc_on_traces(udr1, traces, [os.path.join(
    #     #     save_dir, "udr1", "step_{}".format(step)) for save_dir in save_dirs])
    #     # test_cc_on_traces(udr2, traces, [os.path.join(
    #     #     save_dir, "udr2", "step_{}".format(step)) for save_dir in save_dirs])
    #     # test_cc_on_traces(udr3, traces, [os.path.join(
    #     #     save_dir, "udr3", "step_{}".format(step)) for save_dir in save_dirs])
    #     cnt += 1
    n_proc = mp.cpu_count() // 2
    # arguments = [(os.path.join(UDR_ROOT, 'udr_large'.format(
    #     seed), "seed_{}".format(seed)), traces, save_dirs, 'udr3', seed) for seed in range(10, 60, 10)]
    # with mp.Pool(processes=n_proc) as pool:
    #     pool.starmap(run_udr_model, arguments)
    # arguments = [(os.path.join(UDR_ROOT, 'udr_small'.format(
    #     seed), "seed_{}".format(seed)), traces, save_dirs, 'udr1', seed) for seed in range(10, 60, 10)]
    # with mp.Pool(processes=n_proc) as pool:
    #     pool.starmap(run_udr_model, arguments)
    arguments = [(os.path.join(UDR_ROOT, 'udr_mid', "seed_{}".format(seed)),
        traces, save_dirs, 'udr2', seed) for seed in range(10, 60, 10)]
    with mp.Pool(processes=n_proc) as pool:
        pool.starmap(run_udr_model, arguments)

    # run_genet_model(GENET_ROOT, traces, save_dirs, seed=20)



if __name__ == "__main__":
    main()
