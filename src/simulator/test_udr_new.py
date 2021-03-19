import argparse
import itertools
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.utils import read_json_file


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Test UDR models in simulator.")
    # parser.add_argument('--exp-name', type=str, default="",
    #                     help="Experiment name.")
    parser.add_argument("--model-path", type=str, default=[], nargs="+",
                        help="Path to one or more Aurora Tensorflow checkpoints"
                        " or models to serve.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to save the model.")
    parser.add_argument('--dimension', type=str,  # nargs=1,
                        choices=["bandwidth", "delay", "loss", "queue"])
    parser.add_argument('--config-file', type=str, required=True,
                        help="config file")
    parser.add_argument('--duration', type=int, required=True,
                        help="trace duration")
    parser.add_argument('--delta-scale', type=float, required=True,
                        help="delta scale")
    parser.add_argument('--plot-only', action='store_true',
                        help='plot only if specified')
    return parser.parse_args()


def learnability_objective_function(throughput, delay):
    """Objective function used in https://cs.stanford.edu/~keithw/www/Learnability-SIGCOMM2014.pdf
    throughput: Mbps
    delay: ms
    """
    score = np.log(throughput) - np.log(delay)
    # print(throughput, delay, score)
    score = score.replace([np.inf, -np.inf], np.nan).dropna()

    return score


def pcc_aurora_reward(throughput, delay, loss):
    """PCC Aurora reward.
    throughput: packets per second
    delay: second
    loss:
    """
    return 10 * throughput - 1000 * delay - 2000 * loss


def main():
    args = parse_args()
    metric = args.dimension
    model_paths = args.model_path
    save_root = args.save_dir
    config_file = args.config_file
    print(metric, model_paths, save_root, config_file)

    plt.figure()
    config = read_json_file(config_file)
    print(config)
    bw_list = config['bandwidth']
    delay_list = config['delay']
    loss_list = config['loss']
    queue_list = config["queue"]

    metric_list_dict = {
        "bandwidth": config['bandwidth'],
        "delay": config['delay'],
        'loss': config['loss'],
        "queue": config["queue"]
    }

    for model_idx, (model_path, ls, marker, color) in enumerate(
            zip(model_paths, ["-", "--", "-.", "-", "-", "-", "-", "-"],
                ["x", "s", "v", '*', '+', '^', '>', '1'],
                ["C3", "C3", "C3", "C1", "C1", "C1", "C1", "C1"])):
        model_name = os.path.basename(os.path.dirname(model_path))
        aurora_rewards = []
        cubic_rewards = []
        aurora_scores = []
        cubic_scores = []

        for (bw, delay, loss, queue) in itertools.product(
                bw_list, delay_list, loss_list, queue_list):
            # print(bw, delay, loss, queue)
            save_dir = f"{save_root}/rand_{metric}/env_{bw}_{delay}_{loss}_{queue}"
            aurora_save_dir = os.path.join(save_dir, model_name)
            cubic_save_dir = os.path.join(save_dir, "cubic")
            # print(save_dir, aurora_save_dir, cubic_save_dir)

            # run aurora
            if not args.plot_only:
                cmd = "CUDA_VISIBLE_DEVICES='' python workspace/evaluate_aurora.py " \
                    "--bandwidth {} --delay {} --loss {} --queue {} " \
                    "--save-dir {} --model-path {} --duration {} --delta-scale {}".format(
                        bw, delay, loss, queue, aurora_save_dir, model_path,
                        args.duration, args.delta_scale)
                subprocess.check_output(cmd, shell=True).strip()

            df = pd.read_csv(os.path.join(
                aurora_save_dir, "aurora_test_log.csv"))
            aurora_rewards.append(df['reward'].mean())
            aurora_scores.append(np.mean(learnability_objective_function(
                df['throughput'] * 1500 * 8 / 1e6, df['latency'] * 1000/2)))
            if model_idx == 0:
                # run cubic
                if not args.plot_only:
                    cmd = "python evaluate_cubic.py --bandwidth {} --delay {} " \
                        "--loss {} --queue {} --save-dir {} --duration {}".format(
                            bw, delay, loss, queue, cubic_save_dir, args.duration)
                    subprocess.check_output(cmd, shell=True).strip()
                df = pd.read_csv(os.path.join(
                    cubic_save_dir, "cubic_test_log.csv"))
                cubic_rewards.append(df['reward'].mean())
                cubic_scores.append(np.mean(learnability_objective_function(
                    df['throughput'] * 1500 * 8 / 1e6, df['latency']*1000/2)))
                # plt.plot(metric_list_dict[metric], cubic_scores,
                #          'o-', c="C0", label="TCP Cubic")

        if model_idx == 0:
            plt.plot(metric_list_dict[metric], cubic_rewards,
                     'o-', c="C0", label="TCP Cubic")
        # if model_idx == 0 or model_idx == 2:
        #     continue
        plt.plot(metric_list_dict[metric], aurora_rewards, marker=marker,
                 ls=ls, c=color, label="Aurora " + model_name)
        # plt.plot(metric_list_dict[metric], scores, marker=marker,
        #          c="C3", ls=ls, label="Aurora " + model_name)
    plt.legend()
    plt.xlabel(metric)
    plt.ylabel('Reward')
    # plt.ylabel('log(throughput) - log(delay)')
    plt.savefig(os.path.join(
        args.save_dir, "rand_{}_sim.png".format(metric)))
    plt.close()
    # plt.show()


if __name__ == "__main__":
    main()
