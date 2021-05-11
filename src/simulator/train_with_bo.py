# This is the main training setup for training a Pensieve model, but
# stopping at intervals to add new training data based on performance
# (increases generalizability, we hope!)
import argparse
import copy
import glob
import os
import itertools
import subprocess
import sys
import time
from typing import Dict, List

import gym
import ipdb
import numpy as np
from bayes_opt import BayesianOptimization

from common.utils import natural_sort, read_json_file, write_json_file, set_seed
from plot_scripts.plot_packet_log import PacketLog

from simulator.evaluate_cubic import test_on_trace as test_cubic_on_trace
from simulator.aurora import Aurora
from simulator.trace import generate_trace
# from simulator.good_network_sim import RandomizationRanges


MODEL_PATH = ""
DEFAULT_CONFIGS = []


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("BO training in simulator.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to testing results.")
    parser.add_argument('--model-path', type=str, default="",
                        help="path to Aurora model to start from.")
    parser.add_argument("--config-file", type=str,
                        help="Path to configuration file.")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    # parser.add_argument('--duration', type=int, default=10,
    #                     help='Flow duration in seconds.')
    parser.add_argument('--total-timesteps', type=int, default=5e6,
                        help='Total timesteps can be used to train.')
    parser.add_argument('--bo-interval', type=int, default=5e4,
                        help='Number of epoch/step used for training between '
                        'two BOs.')
    # parser.add_argument('--new-range-weight', type=float, default=0.7,
    #                     help='New range weight.')

    return parser.parse_args()


class RandomizationRanges():
    def __init__(self, filename):
        if filename and os.path.exists(filename):
            self.rand_ranges = read_json_file(filename)
        else:
            self.rand_ranges = []

    def add_range(self, range_maps, prob=0.4):
        for rand_range in self.rand_ranges:
            rand_range['weight'] *= (1 - prob)
        # new_range = copy.deepcopy(self.rand_ranges[0])
        if self.rand_ranges:
            weight = prob / len(range_maps)
        else:
            weight = 1 / len(range_maps)
        for range_map in range_maps:
            # for dim in range_map:
            #     if dim not in new_range:
            #         raise RuntimeError
            #     new_range[dim] = range_map[dim]
            range_map['weight'] = weight
            self.rand_ranges.append(range_map)

    def get_original_range(self):
        return self.rand_ranges[0]

    def get_ranges(self):
        return self.rand_ranges

    def dump(self, filename):
        write_json_file(filename, self.rand_ranges)


def map_log_to_lin(x):
    x_lin = 2**(10*x)
    return x_lin


def latest_model_from(path):
    ckpts = list(glob.glob(os.path.join(path, "model_step_*.ckpt.meta")))
    # models_to_serve = list(
    #     glob.glob(os.path.join(path, "model_to_serve_step_*")))
    if not ckpts:
        ckpt = ""
    else:
        ckpt = os.path.splitext(natural_sort(ckpts)[-1])[0]
    if not models_to_serve:
        model_to_serve = ""
    else:
        model_to_serve = natural_sort(models_to_serve)[-1]

    return ckpt, model_to_serve


# def black_box_function(bandwidth):
#     '''Compute reward gap between baseline and RL solution.
#
#     Args
#         delay: One-way link delay (ms).
#
#     Return
#          (cubic - aurora) reward
#     '''
#     assert MODEL_PATH != ""
#     # cmd = "python evaluate_cubic.py --bandwidth {} --delay {} --queue {} " \
#     #     "--loss {} --duration {} --save-dir {}".format(
#     #         bandwidth, delay, int(queue), loss, 10, "tmp")
#     # print(cmd)
#     # subprocess.check_output(cmd, shell=True).strip()
#     # print(DEFAULT_CONFIGS)
#
#     reward_diffs = []
#     for config in DEFAULT_CONFIGS:
#         trace = generate_trace(duration_range=(10, 10), bandwidth_range=(1, bandwidth),
#                                delay_range=(config['delay'], config['delay']),
#                                loss_rate_range=(
#                                    config['loss'], config['loss']),
#                                queue_size_range=(
#                                    config['queue'], config['queue']),
#                                T_s_range=(config['T_s'], config['T_s']),
#                                delay_noise_range=(
#                                    config['delay_noise'], config['delay_noise']),
#                                constant_bw=False)
#         cubic_rewards, cubic_pkt_log = test_cubic_on_trace(trace, "tmp", 20)
#
#         aurora = Aurora(seed=20, log_dir="tmp", timesteps_per_actorbatch=10,
#                         pretrained_model_path=MODEL_PATH, delta_scale=1)
#         ts_list, reward_list, loss_list, tput_list, delay_list, send_rate_list, \
#             action_list, obs_list, mi_list, pkt_log = aurora.test(trace, 'tmp')
#         reward_diffs.append(np.mean(np.array(cubic_rewards)
#                                     ) - np.mean(np.array(reward_list)))
#
#     return np.mean(np.array(reward_diffs))

def black_box_function(delay):
    '''Compute reward gap between baseline and RL solution.

    Args
        delay: One-way link delay (ms).

    Return
         (cubic - aurora) reward
    '''
    assert MODEL_PATH != ""
    # cmd = "python evaluate_cubic.py --bandwidth {} --delay {} --queue {} " \
    #     "--loss {} --duration {} --save-dir {}".format(
    #         bandwidth, delay, int(queue), loss, 10, "tmp")
    # print(cmd)
    # subprocess.check_output(cmd, shell=True).strip()
    # print(DEFAULT_CONFIGS)

    reward_diffs = []
    for config in DEFAULT_CONFIGS:
        trace = generate_trace(duration_range=(10, 10), bandwidth_range=(1, config['bandwidth']),
                               delay_range=(delay, delay),
                               loss_rate_range=(
                                   config['loss'], config['loss']),
                               queue_size_range=(
                                   config['queue'], config['queue']),
                               T_s_range=(config['T_s'], config['T_s']),
                               delay_noise_range=(
                                   config['delay_noise'], config['delay_noise']),
                               constant_bw=False)
        cubic_rewards, cubic_pkt_log = test_cubic_on_trace(trace, "tmp", 20)

        aurora = Aurora(seed=20, log_dir="tmp", timesteps_per_actorbatch=10,
                        pretrained_model_path=MODEL_PATH, delta_scale=1)
        ts_list, reward_list, loss_list, tput_list, delay_list, send_rate_list, \
            action_list, obs_list, mi_list, pkt_log = aurora.test(trace, 'tmp')
        reward_diffs.append(PacketLog.from_log(cubic_pkt_log).get_reward(None, trace=trace)
                - PacketLog.from_log(pkt_log).get_reward(None, trace=trace))
        # reward_diffs.append(np.mean(np.array(cubic_rewards)
        #                             ) - np.mean(np.array(reward_list)))

    return np.mean(np.array(reward_diffs))

# def black_box_function(T_s):
#     '''Compute reward gap between baseline and RL solution.
#
#     Args
#         delay: One-way link delay (ms).
#
#     Return
#          (cubic - aurora) reward
#     '''
#     assert MODEL_PATH != ""
#     # cmd = "python evaluate_cubic.py --bandwidth {} --delay {} --queue {} " \
#     #     "--loss {} --duration {} --save-dir {}".format(
#     #         bandwidth, delay, int(queue), loss, 10, "tmp")
#     # print(cmd)
#     # subprocess.check_output(cmd, shell=True).strip()
#     # print(DEFAULT_CONFIGS)
#
#     reward_diffs = []
#     for config in DEFAULT_CONFIGS:
#         trace = generate_trace(duration_range=(10, 10), bandwidth_range=(1, config['bandwidth']),
#                                delay_range=(config['delay'], config['delay']),
#                                loss_rate_range=(
#                                    config['loss'], config['loss']),
#                                queue_size_range=(
#                                    config['queue'], config['queue']),
#                                T_s_range=(T_s, T_s),
#                                delay_noise_range=(
#                                    config['delay_noise'], config['delay_noise']),
#                                constant_bw=False)
#         cubic_rewards, cubic_pkt_log = test_cubic_on_trace(trace, "tmp", 20)
#
#         aurora = Aurora(seed=20, log_dir="tmp", timesteps_per_actorbatch=10,
#                         pretrained_model_path=MODEL_PATH, delta_scale=1)
#         ts_list, reward_list, loss_list, tput_list, delay_list, send_rate_list, \
#             action_list, obs_list, mi_list, pkt_log = aurora.test(trace, 'tmp')
#         reward_diffs.append(np.mean(np.array(cubic_rewards)
#                                     ) - np.mean(np.array(reward_list)))
#
#     return np.mean(np.array(reward_diffs))

# def black_box_function(queue):
#     assert MODEL_PATH != ""
#
#     reward_diffs = []
#     for config in DEFAULT_CONFIGS:
#         trace = generate_trace(duration_range=(10, 10), bandwidth_range=(1, config['bandwidth']),
#                                delay_range=(config['delay'], config['delay']),
#                                loss_rate_range=(
#                                    config['loss'], config['loss']),
#                                queue_size_range=(queue, queue),
#                                T_s_range=(config['T_s'], config['T_s']),
#                                delay_noise_range=(
#                                    config['delay_noise'], config['delay_noise']),
#                                constant_bw=False)
#         cubic_rewards, cubic_pkt_log = test_cubic_on_trace(trace, "tmp", 20)
#
#         aurora = Aurora(seed=20, log_dir="tmp", timesteps_per_actorbatch=10,
#                         pretrained_model_path=MODEL_PATH, delta_scale=1)
#         ts_list, reward_list, loss_list, tput_list, delay_list, send_rate_list, \
#             action_list, obs_list, mi_list, pkt_log = aurora.test(trace, 'tmp')
#         reward_diffs.append(np.mean(np.array(cubic_rewards)
#                                     ) - np.mean(np.array(reward_list)))
#
#     return np.mean(np.array(reward_diffs))

def select_range_from_opt_res(res):
    """Select parameter range from BO optimizer res."""
    params = list(res[0]['params'].keys())
    targets = [val['target'] for val in res]
    selected_ranges = {}
    pos_target_indices = [idx for idx,
                          target in enumerate(targets) if target > 0]
    selected_indices = pos_target_indices[:3] if pos_target_indices else np.argsort(
        np.array(targets))[:5]

    for param in params:
        values = []
        for idx in selected_indices:
            values.append(res[idx]['params'][param])
        selected_ranges[param] = [min(values), max(values)]
    return selected_ranges


def do_bo(pbounds, black_box_function, seed):
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=seed
    )

    optimizer.maximize(init_points=13, n_iter=2, kappa=20, xi=0.1)
    best_param = optimizer.max
    # delay = next.get('params').get('delay')
    # queue = next.get('params').get('queue')
    # param_TS = next.get('params').get('y')
    # params_range = select_range_from_opt_res(optimizer.res)
    return best_param  # , params_range


def initalize_pretrained_model(pbounds: Dict[str, List], n_models: int, save_dir: str,
                               duration: int, steps: int):
    """Initialize a pretrained model for BO training."""
    # processes = []
    # for idx, (bw, delay, loss, queue) in enumerate(zip(
    #         np.linspace(pbounds['bandwidth'][0],
    #                     pbounds['bandwidth'][1], n_models),
    #         np.linspace(pbounds['delay'][0], pbounds['delay'][1], n_models),
    #         np.linspace(pbounds['loss'][0], pbounds['loss'][1], n_models),
    #         np.linspace(pbounds['queue'][0], pbounds['queue'][1], n_models))):
    #     cmd = "python train_rl.py \
    #         --save-dir {save_dir} \
    #         --exp-name {exp_name} \
    #         --duration {duration} \
    #         --tensorboard-log aurora_tensorboard \
    #         --total-timesteps {steps} \
    #         --bandwidth {min_bw} {max_bw} \
    #         --delay {min_delay} {max_delay}  \
    #         --loss {min_loss} {max_loss} \
    #         --queue {min_queue} {max_queue} \
    #         --delta-scale 1".format(
    #         save_dir=os.path.join(save_dir, 'pretrained_model_{}'.format(idx)),
    #         exp_name='{}_pretrained_model_{}'.format(
    #             os.path.basename(save_dir), idx), duration=duration,
    #         steps=steps, min_bw=bw, max_bw=bw,
    #         min_delay=delay, max_delay=delay, min_loss=loss, max_loss=loss,
    #         min_queue=round(queue), max_queue=round(queue))
    #     processes.append(subprocess.Popen(
    #         cmd.split(),
    #         stdout=open(os.path.join(save_dir,
    #                                  'pretrained_model_{}'.format(idx),
    #                                  'stdout.log'), 'w', 1),
    #         stderr=open(os.path.join(save_dir,
    #                                  'pretrained_model_{}'.format(idx),
    #                                  'stderr.log'), 'w', 1)))
    #
    # while True:
    #     for p in processes:
    #         if p.poll() is not None:
    #             if p.returncode == 0:
    #                 processes.remove(p)
    #             else:
    #                 raise RuntimeError(p.args + 'failed!')
    #     if not processes:
    #         break
    #     else:
    #         time.sleep(10)

    # TODO: do a grid sweeping here.

    for idx, (bw, delay, loss, queue) in enumerate(zip(
            np.linspace(pbounds['bandwidth'][0],
                        pbounds['bandwidth'][1], n_models),
            np.linspace(pbounds['delay'][0], pbounds['delay'][1], n_models),
            np.linspace(pbounds['loss'][0], pbounds['loss'][1], n_models),
            np.linspace(pbounds['queue'][0], pbounds['queue'][1], n_models))):
        ckpt_path, model2serve_path = latest_model_from(
            os.path.join(save_dir, 'pretrained_model_{}'.format(idx)))
        return ckpt_path
        # print(model2serve_path)
        # cmd = "CUDA_VISIBLE_DEVICES='' python evaluate_aurora.py --bandwidth {} --delay {} --queue {} "  \
        #     "--loss {} --model-path {} --duration {}".format(
        #         bandwidth, delay, int(queue), loss, MODEL_PATH, 10)


def main():
    set_seed(20)
    global MODEL_PATH
    global DEFAULT_CONFIGS
    args = parse_args()

    print(args)
    for _ in range(10):
        DEFAULT_CONFIGS.append(
            {
             "bandwidth": 10 ** np.random.uniform(np.log10(1), np.log10(6), 1).item(),
             # "delay": np.random.uniform(5, 200, 1).item(),
             "loss": np.random.uniform(0, 0.0, 1).item(),
             "queue": 10 ** np.random.uniform(np.log10(2), np.log10(30), 1).item(),
             "T_s": np.random.randint(0, 6, 1).item(),
             "delay_noise": np.random.uniform(0, 0, 1).item()})
    if args.model_path:
        # model path is specified, so use it as the pretrained model
        MODEL_PATH = args.model_path
    else:
        # model path is not specified, strategies to create a model to start
        # strategy 1: start from a randomly initialized model

        # strategy 2: randomly sample input ranges of all dimensions and
        # construct an environments with the samplings. Train 1 model on the
        # environment and use it as the model to start.

        # strategy 3: sample input ranges of all dimensions and construct the
        # environments with the samplings. Train n models on the environments.
        # Then test then by grid-sweeping the parameter space.
        # pick the one which has the highest reward to be the model to start.
        MODEL_PATH = ""
    config_file = args.config_file
    # config = read_json_file(args.config_file)[0]
    # pbounds = {k: config[k] for k in config if k != 'weight'}
    pbounds = {'delay': (5, 200)}
    # pbounds = {'T_s': (0, 6)}
    # pbounds = {'queue': (2, 200)}
    # pbounds = {'bandwidth': (1, 6)}
    # ckpt_path = initalize_pretrained_model(pbounds, 5, "test_bo", 10, 10000)
    # ckpt_path = "../../results_0415/udr_7_dims/range0/model_step_14400.ckpt"
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    # Example Flow:
    for i in range(12):
        best_param = do_bo(pbounds, black_box_function, seed=args.seed)
        # min_bw, max_bw = params_range['bandwidth']
        # min_delay, max_delay = params_range['delay']
        # min_loss, max_loss = params_range['loss']
        # min_queue, max_queue = params_range['queue']

        print("BO choose best paramter", best_param)
        range_maps = []
        for config in DEFAULT_CONFIGS:
            # range_maps.append(
            #     {'bandwidth': [1, best_param['params']['bandwidth']],
            #      'delay': [config['delay'], config['delay']],
            #      'loss': [config['loss'], config['loss']],
            #      'queue': [config['queue'], config['queue']],
            #      'T_s': [config['T_s'], config['T_s']],
            #      'delay_noise': [config['delay_noise'],
            #                      config['delay_noise']],
            #      })
            range_maps.append(
                {'bandwidth': [1, config['bandwidth']],
                 'delay': [best_param['params']['delay'], best_param['params']['delay']],
                 'loss': [config['loss'], config['loss']],
                 'queue': [config['queue'], config['queue']],
                 'T_s': [config['T_s'], config['T_s']],
                 'delay_noise': [config['delay_noise'],
                                 config['delay_noise']],
                 })
            # range_maps.append(
            #     {'bandwidth': [1, config['bandwidth']],
            #      'delay': [config['delay'], config['delay']],
            #      'loss': [config['loss'], config['loss']],
            #      'queue': [best_param['params']['queue'], best_param['params']['queue']],
            #      'T_s': [config['T_s'], config['T_s']],
            #      'delay_noise': [config['delay_noise'],
            #                      config['delay_noise']],
            #      })
            # range_maps.append(
            #     {'bandwidth': [1, config['bandwidth']],
            #      'delay': [config['delay'], config['delay']],
            #      'loss': [config['loss'], config['loss']],
            #      'queue': [config['queue'], config['queue']],
            #      'T_s': [best_param['params']['T_s'], best_param['params']['T_s']],
            #      'delay_noise': [config['delay_noise'],
            #                      config['delay_noise']],
            #      })

        rand_ranges = RandomizationRanges(config_file)
        rand_ranges.add_range(range_maps)
        new_config_file = os.path.join(args.save_dir, "bo_"+str(i) + ".json")
        rand_ranges.dump(new_config_file)
        config_file = new_config_file

        # Use the new param, add more traces into Pensieve, train more based on
        # before
        # latest_ckpt_path, latest_model_path = latest_model_from(save_dir)

        # train Aurora with new parameter
        command = "mpirun -np 2 python train_rl.py " \
            "--seed {seed} --delta-scale 1 --time-variant-bw --total-timesteps {tot_step} " \
            "--save-dir {save_dir}/bo_{i} " \
            "--pretrained-model-path {model_path} " \
            "--randomization-range-file {config_file}".format(
                seed=args.seed, i=i,
                tot_step=36000, save_dir=save_dir,
                model_path=MODEL_PATH,
                config_file=config_file)
        print(command)
        subprocess.check_call(command.split())

        command = "python ../plot_scripts/plot_validation.py --log-file " \
                  "{save_dir}/validation_log.csv --save-dir {save_dir}".format(
                      save_dir="{}/bo_{}".format(save_dir, i))
        print(command)
        MODEL_PATH = subprocess.check_output(command.split()).strip().decode("utf-8")
        print(MODEL_PATH)

        # MODEL_PATH, latest_model_path = latest_model_from("{}/bo_{}".format())
        # print('MODEL_PATH changes to', MODEL_PATH)
        # os.system("rm -r /tmp/best_model_to_serve")
        # print("cp -r {} /tmp/best_model_to_serve".format(latest_model_path))
        # os.system("cp -r {} /tmp/best_model_to_serve".format(latest_model_path))
        #
        # print("Get the file and pass it to the training script, if it exists.\n")
        # print("Running training:", i)


if __name__ == "__main__":
    main()
