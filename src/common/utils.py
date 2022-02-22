import csv
import json
import logging
import os
import random
import re
from typing import Union, Dict

import numpy as np


def read_json_file(filename):
    """Load json object from a file."""
    with open(filename, 'r') as f:
        content = json.load(f)
    return content


def write_json_file(filename, content):
    """Dump into a json file."""
    with open(filename, 'w') as f:
        json.dump(content, f, indent=4)


def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def learnability_objective_function(throughput, delay):
    """Objective function used in https://cs.stanford.edu/~keithw/www/Learnability-SIGCOMM2014.pdf
    throughput: Mbps
    delay: ms
    """
    score = np.log(throughput) - np.log(delay)
    # print(throughput, delay, score)
    score = score.replace([np.inf, -np.inf], np.nan).dropna()

    return score


def pcc_aurora_reward(throughput: float, delay: float, loss: float,
                      avg_bw: Union[float, None] = None,
                      min_rtt: Union[float, None] = None) -> float:
    """PCC Aurora reward. Anchor point 0.6Mbps
    throughput: packets per second
    delay: second
    loss:
    avg_bw: packets per second
    """
    # if avg_bw is not None and min_rtt is not None:
    #     return 10 * 50 * throughput/avg_bw - 1000 * delay * 0.2 / min_rtt - 2000 * loss
    if avg_bw is not None:
        return 10 * 50 * throughput/avg_bw - 1000 * delay - 2000 * loss
    return 10 * throughput - 1000 * delay - 2000 * loss


def compute_std_of_mean(data):
    return np.std(data) / np.sqrt(len(data))


def load_summary(summary_file: str) -> Dict[str, float]:
    summary = {}
    with open(summary_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                summary[k] = float(v)
    return summary


def save_args(args, save_dir: str):
    """Write arguments to a log file."""
    os.makedirs(save_dir, exist_ok=True)
    if save_dir and os.path.exists(save_dir):
        write_json_file(os.path.join(save_dir, 'cmd.json'), args.__dict__)


def zero_one_normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def load_bo_json_log(file: str):
    logs = []
    with open(file, 'r') as f:
        for line in f:
            line.strip()
            logs.append(json.loads(line))
    return logs
