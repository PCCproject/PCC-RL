import argparse
import csv
import json
import os
import time
from typing import Callable, Dict, List, Set, Union

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from mpi4py.MPI import COMM_WORLD

from common.utils import (pcc_aurora_reward, read_json_file, set_seed, write_json_file)
from simulator.aurora import Aurora, test_on_traces
from simulator.network_simulator.constants import BITS_PER_BYTE, BYTES_PER_PACKET
from simulator.network_simulator.cubic import Cubic
from simulator.network_simulator.bbr import BBR
from simulator.trace import generate_trace

MODEL_PATH = ""


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("BO training in simulator.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="directory to save testing and intermediate results.")
    parser.add_argument('--model-path', type=str, default=None,
                        help="path to Aurora model to start from.")
    parser.add_argument("--config-file", type=str,
                        help="Path to configuration file.")
    parser.add_argument("--bo-rounds", type=int, default=30,
                        help="Rounds of BO.")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--heuristic', type=str, default="cubic",
                        choices=('bbr', 'cubic', 'optimal'),
                        help='Congestion control rule based method.')

    return parser.parse_args()


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


# class BasicObserver:
#     def update(self, event, instance):
#         """Does whatever you want with the event and `BayesianOptimization` instance."""
#         print("Event `{}` was observed".format(event))


class Genet:
    """Genet implementation with Bayesian Optimization.

    Args
        config_file: configuration file which contains the large ranges of all parameters.
        black_box_function: a function to maximize reward_diff = heuristic_reward - rl_reward.
        heuristic: an object which is the abstraction of a rule-based method.
        rl_method: an object which is the abstraction of a RL-based method.
        seed: random seed.
    """

    def __init__(self, config_file: str, save_dir: str,
                 black_box_function: Callable, heuristic, rl_method,
                 seed: int = 42):
        self.black_box_function = black_box_function
        self.seed = seed
        self.config_file = config_file
        self.cur_config_file = config_file
        self.rand_ranges = RandomizationRanges(config_file)
        self.param_names = self.rand_ranges.get_parameter_names()
        self.pbounds = self.rand_ranges.get_original_range()
        if 'duration' in self.pbounds:
            self.pbounds.pop('duration')

        self.save_dir = save_dir
        self.heuristic = heuristic
        self.rl_method = rl_method
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
            if COMM_WORLD.Get_rank() == 0:
                optimizer = BayesianOptimization(
                    f=lambda bandwidth_lower_bound, bandwidth_upper_bound, delay,
                    queue, loss, T_s, delay_noise: self.black_box_function(
                        bandwidth_lower_bound, bandwidth_upper_bound, delay, queue,
                        loss, T_s, delay_noise, heuristic=self.heuristic,
                        rl_method=self.rl_method),
                    pbounds=self.pbounds, random_state=self.seed+i)
                os.makedirs(os.path.join(self.save_dir, "bo_{}".format(i)),
                            exist_ok=True)
                logger = JSONLogger(path=os.path.join(
                    self.save_dir, "bo_{}_logs.json".format(i)))
                optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
                optimizer.maximize(init_points=10, n_iter=5, kappa=20, xi=0.1)
                best_param = optimizer.max
                print(best_param)
                self.rand_ranges.add_ranges([best_param['params']])
                self.cur_config_file = os.path.join(
                    self.save_dir, "bo_"+str(i) + ".json")
                self.rand_ranges.dump(self.cur_config_file)
                to_csv(self.cur_config_file)
                self.rl_method.log_dir = os.path.join(self.save_dir, "bo_{}".format(i))
                data = self.cur_config_file
                # for i in range(1, COMM_WORLD.Get_size()):
                #     COMM_WORLD.send(self.cur_config_file, dest=i, tag=11)
            else:
                data = None # self.cur_config_file = COMM_WORLD.recv(source=0, tag=11)
            self.cur_config_file = COMM_WORLD.bcast(data, root=0)
            print(COMM_WORLD.Get_rank(), self.cur_config_file)
            self.rl_method.train(self.cur_config_file, 7.2e4, 100)

def black_box_function(bandwidth_lower_bound: float,
                       bandwidth_upper_bound: float, delay: float, queue: float,
                       loss: float, T_s: float, delay_noise: float, heuristic, rl_method) -> float:
    if COMM_WORLD.Get_rank() == 0:
        model_path = rl_method.save_model(rl_method.log_dir)
        heuristic_rewards = []
        rl_method_rewards = []
        traces = [generate_trace(
            duration_range=(30, 30),
            bandwidth_lower_bound_range=(bandwidth_lower_bound, bandwidth_lower_bound),
            bandwidth_upper_bound_range=(bandwidth_upper_bound, bandwidth_upper_bound),
            delay_range=(delay, delay),
            loss_rate_range=(loss, loss),
            queue_size_range=(queue, queue),
            T_s_range=(T_s, T_s),
            delay_noise_range=(delay_noise, delay_noise)) for _ in range(10)]
            # print("trace generation used {}s".format(time.time() - t_start))
        if not heuristic:
            for trace in traces:
                heuristic_rewards.append(pcc_aurora_reward(
                trace.avg_bw * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET,
                trace.avg_delay * 2 / 1000, trace.loss_rate, trace.avg_bw))
        else:
            t_start = time.time()
            # heuristic_mi_level_reward, heuristic_pkt_level_reward = heuristic.test(trace, "")
            hret = heuristic.test_on_traces(traces, [""]* len(traces) , False, COMM_WORLD.Get_size())
            for heuristic_mi_level_reward, heuristic_pkt_level_reward in hret:
                heuristic_rewards.append(heuristic_mi_level_reward)
            print("heuristic used {}s".format(time.time() - t_start))
            t_start = time.time()
            # commented code buggy: run out of file descriptor.
            # rl_ret = test_on_traces(model_path, traces, [rl_method.log_dir] * len(traces), COMM_WORLD.Get_size(), 20)
            # for rl_mi_level_reward, rl_pkt_level_reward in rl_ret:
            #     rl_method_rewards.append(rl_mi_level_reward)
            for trace in traces:
                rl_mi_level_reward, rl_pkt_level_reward = rl_method.test(trace, rl_method.log_dir)
                rl_method_rewards.append(rl_mi_level_reward)
            print("rl used {}s".format(time.time() - t_start))
            # print(rl_method_rewards)
            # rl_method_rewards.append(rl_pkt_level_reward)
        gap = float(np.mean(heuristic_rewards) - np.mean(rl_method_rewards))
        return gap
    raise RuntimeError("Should not reach here.")


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
        writer = csv.DictWriter(
                f, ['bandwidth_lower_bound', 'bandwidth_upper_bound', 'delay',
                    'queue', 'loss', 'T_s', "delay_noise", 'duration', 'weight'])
        writer.writeheader()
        for config in bo_log:
            writer.writerow(config)


def main():
    args = parse_args()
    set_seed(args.seed + COMM_WORLD.Get_rank())

    if args.heuristic == 'bbr':
        heuristic = BBR(True)
    elif args.heuristic == 'cubic':
        heuristic = Cubic(True)
    elif args.heuristic == 'optimal':
        heuristic = None
    else:
        raise ValueError
    nprocs = COMM_WORLD.Get_size()
    aurora = Aurora(seed=args.seed, log_dir=args.save_dir,
                    pretrained_model_path=args.model_path,
                    timesteps_per_actorbatch=int(7200/nprocs), delta_scale=1)
    genet = Genet(args.config_file, args.save_dir, black_box_function,
                  heuristic, aurora)
    genet.train(args.bo_rounds)


if __name__ == "__main__":
    main()
