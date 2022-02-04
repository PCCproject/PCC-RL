import argparse
import copy
import csv
import glob
import json
import os
import subprocess
import time
import warnings
from typing import Callable, Dict, List, Set, Union

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.observer import _Tracker
from bayes_opt.event import Events
import pandas as pd

from common.utils import (
    natural_sort, read_json_file, set_seed, write_json_file)
from simulator.aurora import test_on_traces
from simulator.network_simulator.cubic import Cubic
from simulator.network_simulator.bbr import BBR
from simulator.network_simulator.bbr_old import BBR_old
from simulator.trace import generate_trace

black_box_function_calling_times = 0


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("BO training in simulator.")
    parser.add_argument('--save-dir', type=str, required=True, help="directory"
                        " to save testing and intermediate results.")
    parser.add_argument('--model-path', type=str, default=None,
                        help="path to Aurora model to start from.")
    parser.add_argument("--config-file", type=str, required=True,
                        help="Path to configuration file.")
    parser.add_argument("--bo-rounds", type=int, default=30,
                        help="Rounds of BO.")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--heuristic', type=str, default="cubic",
                        choices=('bbr', 'bbr_old', 'cubic', 'optimal'),
                        help='Congestion control rule based method.')
    parser.add_argument('--nproc', type=int, default=2, help='number of '
                        'workers used in training.')
    parser.add_argument('--validation', action='store_true',
                        help='specify to enable validation.')
    parser.add_argument('--n-init-pts', type=int, default=10,
                        help='number of randomly points in BO.')
    parser.add_argument('--n-iter', type=int, default=5,
                        help='number of exploitation points in BO.')
    parser.add_argument('--model-select', type=str, choices=('best', 'latest'),
                        default='latest', help='method to select a model from '
                        'the saved ones.')
    parser.add_argument("--train-trace-file", type=str, default=None,
                        help="A file contains a list of paths to the real "
                        "network traces used in training.")
    parser.add_argument("--real-trace-prob", type=float, default=0,
                        help="Probability of picking a real trace in training")
    parser.add_argument('--bo-only', action='store_true',
                        help='specify to avoid training.')

    return parser.parse_args()


class JSONLogger(_Tracker):
    def __init__(self, path):
        self._path = path if path[-5:] == ".json" else path + ".json"
        try:
            os.remove(self._path)
        except OSError:
            pass
        super(JSONLogger, self).__init__()

    def update(self, event, instance):
        if event == Events.OPTIMIZATION_STEP:
            data = dict(instance.res[-1])

            now, time_elapsed, time_delta = self._time_metrics()
            data["datetime"] = {
                "datetime": now,
                "elapsed": time_elapsed,
                "delta": time_delta,
            }
            data2dump = copy.deepcopy(data)
            data2dump['params']['bandwidth_lower_bound'] = 10**data2dump['params']['bandwidth_lower_bound']
            data2dump['params']['bandwidth_upper_bound'] = 10**data2dump['params']['bandwidth_upper_bound']

            if data2dump['params']['loss'] < -4:
                data2dump['params']['loss'] = 0
            else:
                data2dump['params']['loss'] = 10**data2dump['params']['loss']

            with open(self._path, "a") as f:
                f.write(json.dumps(data2dump) + "\n")

        self._update_tracker(event, instance)


# TODO: add more new configuraiton adding policies
class RandomizationRanges:
    """Manage randomization ranges used in GENET training."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        if filename and os.path.exists(filename):
            self.rand_ranges = read_json_file(filename)
            assert isinstance(self.rand_ranges, List) and len(
                self.rand_ranges) >= 1, "rand_ranges object should be a list with length at least 1."
            weight_sum = 0
            for rand_range in self.rand_ranges:
                weight_sum += rand_range['weight']
            assert weight_sum == 1.0, "Weight sum should be 1."
            self.parameters = set(self.rand_ranges[0].keys())
            self.parameters.remove('weight')
        else:
            self.rand_ranges = []
            self.parameters = set()

    def add_ranges(self, range_maps: List[Dict[str, Union[List[float], float]]],
                   prob: float = 0.3) -> None:
        """Add a list of ranges into the randomization ranges.

        The sum of weights of newlly added ranges is prob.
        """
        for rand_range in self.rand_ranges:
            rand_range['weight'] *= (1 - prob)
        if self.rand_ranges:
            weight = prob / len(range_maps)
        else:
            weight = 1 / len(range_maps)
        for range_map in range_maps:
            range_map_to_add = dict()
            for param in self.parameters:
                if param == 'duration':
                    range_map_to_add[param] = [30, 30]
                    continue

                assert param in range_map, "range_map does not contain '{}'".format(
                    param)
                if param == 'bandwidth_lower_bound' or param == 'bandwidth_upper_bound':
                    range_map_to_add[param] = [10**range_map[param], 10**range_map[param]]
                elif param == 'loss':
                    if range_map[param] < -4:
                        loss = 0
                    else:
                        loss = 10**range_map[param]
                    range_map_to_add[param] = [loss, loss]
                else:
                    range_map_to_add[param] = [range_map[param], range_map[param]]
            range_map_to_add['weight'] = weight
            self.rand_ranges.append(range_map_to_add)

    def get_original_range(self) -> Dict[str, List[float]]:
        start_range = dict()
        for param_name in self.parameters:
            start_range[param_name] = self.rand_ranges[0][param_name]
        return start_range

    def get_ranges(self) -> List[Dict[str, List[float]]]:
        return self.rand_ranges

    def get_parameter_names(self) -> Set[str]:
        return self.parameters

    def dump(self, filename: str) -> None:
        write_json_file(filename, self.rand_ranges)


def get_model_from(path: str, opt='latest') -> str:
    if opt == 'latest':
        ckpts = list(glob.glob(os.path.join(path, "model_step_*.ckpt.meta")))
        if not ckpts:
            ckpt = ""
        else:
            ckpt = os.path.splitext(natural_sort(ckpts)[-1])[0]

        return ckpt
    elif opt == 'best':
        df = pd.read_csv(os.path.join(path, "validation_log.csv"), sep='\t')
        assert isinstance(df, pd.DataFrame)
        best_idx = df['mean_validation_reward'].argmax()
        best_step = int(df['num_timesteps'][best_idx])
        best_ckpt = os.path.join(path, "model_step_{}.ckpt".format(best_step))

        return best_ckpt
    raise ValueError


class Genet:
    """Genet implementation with Bayesian Optimization.

    Args
        config_file: configuration file which contains the large ranges of all parameters.
        save_dir: a path to save results.
        black_box_function: a function to maximize reward_diff = heuristic_reward - rl_reward.
        heuristic: an object which is the abstraction of a rule-based method.
        model_path: path to a starting model.
        nproc: number of processes used in training.
        seed: random seed.
        validation: a boolean flag to enable validation in training.
        n_init_pts: number of randomly sampled points in BO.
        n_iter: number of exploitation points in BO.
        model_select: how to pick a model from trained models. latest or best. Default: latest
    """

    def __init__(self, config_file: str, save_dir: str,
                 black_box_function: Callable, heuristic, model_path: str,
                 nproc: int, seed: int = 42, validation: bool = False,
                 n_init_pts: int = 10, n_iter: int = 5,
                 model_select: str = 'latest',
                 train_trace_file: Union[None, str] = None,
                 real_trace_prob: float = 0, bo_only: bool = False):
        self.real_trace_prob = real_trace_prob
        self.black_box_function = black_box_function
        self.seed = seed
        self.config_file = config_file
        self.cur_config_file = config_file
        self.rand_ranges = RandomizationRanges(config_file)
        self.param_names = self.rand_ranges.get_parameter_names()
        self.pbounds = copy.deepcopy(self.rand_ranges.get_original_range())
        if 'duration' in self.pbounds:
            self.pbounds.pop('duration')
        if 'bandwidth_lower_bound' in self.pbounds:
            self.pbounds['bandwidth_lower_bound'][0] = np.log10(self.pbounds['bandwidth_lower_bound'][0])
            self.pbounds['bandwidth_lower_bound'][1] = np.log10(self.pbounds['bandwidth_lower_bound'][1])
        if 'bandwidth_upper_bound' in self.pbounds:
            self.pbounds['bandwidth_upper_bound'][0] = np.log10(self.pbounds['bandwidth_upper_bound'][0])
            self.pbounds['bandwidth_upper_bound'][1] = np.log10(self.pbounds['bandwidth_upper_bound'][1])

        if 'loss' in self.pbounds:
            self.pbounds['loss'][0] = np.log10(self.pbounds['loss'][0] + 1e-5)
            self.pbounds['loss'][1] = np.log10(self.pbounds['loss'][1] + 1e-5)
        self.save_dir = save_dir
        self.heuristic = heuristic
        self.model_path = model_path  # keep track of the latest model path
        self.start_model_path = model_path
        self.nproc = nproc
        self.validation = validation
        self.n_init_pts = n_init_pts
        self.n_iter = n_iter
        self.bo_only = bo_only
        if model_select != 'latest' and model_select != 'best':
            raise ValueError('Wrong way of model_select!')
        self.model_select = model_select
        self.train_trace_file = train_trace_file
        # my_observer = BasicObserver()
        # self.optimizer.subscribe(
        #     event=Events.OPTIMIZATION_STEP,
        #     subscriber=my_observer,
        #     callback=None)

    def train(self, rounds: int):
        """Genet trains rl_method.
        Args
            rounds: rounds of BO.
        """
        for i in range(rounds):
            training_save_dir = os.path.join(self.save_dir, "bo_{}".format(i))
            optimizer = BayesianOptimization(
                f=lambda bandwidth_lower_bound, bandwidth_upper_bound, delay,
                queue, loss, T_s, delay_noise: self.black_box_function(
                    bandwidth_lower_bound, bandwidth_upper_bound, delay, queue,
                    loss, T_s, delay_noise, heuristic=self.heuristic,
                    model_path=self.model_path,
                    save_dir=os.path.join(training_save_dir, 'bo_traces')),
                pbounds=self.pbounds, random_state=self.seed+i)
            os.makedirs(os.path.join(self.save_dir, "bo_{}".format(i)),
                        exist_ok=True)
            logger = JSONLogger(path=os.path.join(
                self.save_dir, "bo_{}_logs.json".format(i)))
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            optimizer.maximize(init_points=self.n_init_pts, n_iter=self.n_iter,
                               kappa=20, xi=0.1)
            best_param = optimizer.max
            print(best_param)
            self.rand_ranges.add_ranges([best_param['params']])
            self.cur_config_file = os.path.join(
                self.save_dir, "bo_"+str(i) + ".json")
            self.rand_ranges.dump(self.cur_config_file)
            to_csv(self.cur_config_file)
            if self.bo_only:
                return

            cmd = "mpiexec -np {nproc} python train_rl.py " \
                "--save-dir {save_dir} --exp-name {exp_name} --seed {seed} " \
                "--total-timesteps {tot_step} " \
                "--randomization-range-file {config_file} " \
                "--real-trace-prob {real_trace_prob}".format(
                    nproc=self.nproc, save_dir=training_save_dir, exp_name="",
                    seed=self.seed, tot_step=int(7.2e4),
                    config_file=self.cur_config_file,
                    real_trace_prob=self.real_trace_prob)
            if self.model_path:
                " --pretrained-model-path {}".format(self.model_path)
            if self.validation:
                cmd += " --validation"
            if self.train_trace_file:
                cmd += " --train-trace-file {}".format(self.train_trace_file)
            subprocess.run(cmd.split(' '))
            self.model_path = get_model_from(training_save_dir, 'latest')
            print(self.model_path)
            assert self.model_path


def black_box_function(bandwidth_lower_bound: float,
                       bandwidth_upper_bound: float, delay: float, queue: float,
                       loss: float, T_s: float, delay_noise: float, heuristic,
                       model_path: str, save_dir: str = "") -> float:

    global black_box_function_calling_times
    save_dir = os.path.join(save_dir, 'config_{}'.format(
        black_box_function_calling_times % 15))
    black_box_function_calling_times += 1
    heuristic_rewards = []
    rl_method_rewards = []

    if loss < -4:
        loss = 0
    else:
        loss = 10**loss
    traces = [generate_trace(
        duration_range=(30, 30),
        bandwidth_lower_bound_range=(
            10**bandwidth_lower_bound, 10**bandwidth_lower_bound),
        bandwidth_upper_bound_range=(
            10**bandwidth_upper_bound, 10**bandwidth_upper_bound),
        delay_range=(delay, delay),
        loss_rate_range=(loss, loss),
        queue_size_range=(queue, queue),
        T_s_range=(T_s, T_s),
        delay_noise_range=(delay_noise, delay_noise)) for _ in range(10)]
    # print("trace generation used {}s".format(time.time() - t_start))
    save_dirs = [os.path.join(save_dir, 'trace_{}'.format(i)) for i in range(10)]
    if not heuristic:
        for trace in traces:
            heuristic_rewards.append(trace.optimal_reward)
    else:
        t_start = time.time()
        # save_dirs = [""] * len(traces)
        hret = heuristic.test_on_traces(traces, save_dirs, True, 8)
        for heuristic_mi_level_reward, heuristic_pkt_level_reward in hret:
            # heuristic_rewards.append(heuristic_mi_level_reward)
            heuristic_rewards.append(heuristic_pkt_level_reward)
        print("heuristic used {}s".format(time.time() - t_start))
    t_start = time.time()
    rl_ret = test_on_traces(model_path, traces, save_dirs, 8, 20,
                            record_pkt_log=False, plot_flag=True)
    for rl_mi_level_reward, rl_pkt_level_reward in rl_ret:
        # rl_method_rewards.append(rl_mi_level_reward)
        rl_method_rewards.append(rl_pkt_level_reward)
    print("rl used {}s".format(time.time() - t_start))
    gap = float(np.mean(np.array(heuristic_rewards)) - np.mean(np.array(rl_method_rewards)))
    return gap


def read_bo_log(file):
    log = []
    with open(file, 'r') as f:
        for line in f:
            log.append(json.loads(line))
    return log


def to_csv(config_file):
    bo_log = read_json_file(config_file)
    csv_file = os.path.join(
        os.path.dirname(config_file),
        os.path.splitext(os.path.basename(config_file))[0] + ".csv")
    with open(csv_file, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        header = ['bandwidth_lower_bound_min', 'bandwidth_lower_bound_max',
                  'bandwidth_upper_bound_min', 'bandwidth_upper_bound_max',
                  'delay_min', 'delay_max', 'queue_min', 'queue_max', 'loss_min',
                  'loss_max', 'T_s_min', 'T_s_max', 'delay_noise_min',
                  'delay_noise_max', 'duration_min', 'duration_max', 'weight']
        writer.writerow(header)
        for config in bo_log:
            writer.writerow([
                config['bandwidth_lower_bound'][0],
                config['bandwidth_lower_bound'][1],
                config['bandwidth_upper_bound'][0],
                config['bandwidth_upper_bound'][1],
                config['delay'][0], config['delay'][1],
                config['queue'][0], config['queue'][1],
                config['loss'][0], config['loss'][1],
                config['T_s'][0], config['T_s'][1],
                config['delay_noise'][0], config['delay_noise'][1],
                config['duration'][0], config['duration'][1],
                config['weight']])

def save_args(args):
    """Write arguments to a log file."""
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_dir and os.path.exists(args.save_dir):
        write_json_file(os.path.join(args.save_dir, 'cmd.json'), args.__dict__)

def main():
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    args = parse_args()
    save_args(args)
    set_seed(args.seed)

    if args.heuristic == 'bbr':
        heuristic = BBR(False)
    elif args.heuristic == 'bbr_old':
        heuristic = BBR_old(False)
    elif args.heuristic == 'cubic':
        heuristic = Cubic(False)
    elif args.heuristic == 'optimal':
        heuristic = None
    else:
        raise ValueError
    genet = Genet(args.config_file, args.save_dir, black_box_function,
                  heuristic, args.model_path, args.nproc, seed=args.seed,
                  validation=args.validation,
                  n_init_pts=args.n_init_pts, n_iter=args.n_iter,
                  model_select=args.model_select,
                  train_trace_file=args.train_trace_file,
                  real_trace_prob=args.real_trace_prob, bo_only=args.bo_only)
    genet.train(args.bo_rounds)


if __name__ == "__main__":
    main()
