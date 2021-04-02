# This is the main training setup for training a Pensieve model, but
# stopping at intervals to add new training data based on performance
# (increases generalizability, we hope!)
import argparse
import glob
import os
import subprocess
import sys
import time
from typing import Dict, List

import gym
import ipdb
import numpy as np
from bayes_opt import BayesianOptimization
from common.utils import natural_sort, read_json_file, write_json_file

# from simulator.good_network_sim import RandomizationRanges


MODEL_PATH = ""


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("BO training in simulator.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to testing results.")
    parser.add_argument('--model-path', type=str, default="",
                        help="path to Aurora model to start from.")
    parser.add_argument("--config-file", type=str, required=True,
                        help="Path to configuration file.")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--duration', type=int, default=10,
                        help='Flow duration in seconds.')
    parser.add_argument('--total-timesteps', type=int, default=5e6,
                        help='Total timesteps can be used to train.')
    parser.add_argument('--bo-interval', type=int, default=5e4,
                        help='Number of epoch/step used for training between '
                        'two BOs.')
    parser.add_argument('--new-range-weight', type=float, default=0.7,
                        help='New range weight.')

    return parser.parse_args()


class RandomizationRanges():
    def __init__(self, filename):
        self.rand_ranges = read_json_file(filename)

    def add_range(self, bw_range, delay_range, loss_range, queue_range, prob=0.7):
        for rand_range in self.rand_ranges:
            rand_range['weight'] *= 1-prob
        new_range = {'bandwidth': bw_range,
                     'delay': delay_range,
                     'loss': loss_range,
                     'queue': queue_range,
                     'weight': prob}
        self.rand_ranges.append(new_range)

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
    models_to_serve = list(
        glob.glob(os.path.join(path, "model_to_serve_step_*")))
    if not ckpts:
        ckpt = ""
    else:
        ckpt = os.path.splitext(natural_sort(ckpts)[-1])[0]
    if not models_to_serve:
        model_to_serve = ""
    else:
        model_to_serve = natural_sort(models_to_serve)[-1]

    return ckpt, model_to_serve


def black_box_function(bandwidth, delay, loss, queue):
    '''Compute reward gap between baseline and RL solution.

    Args
        delay: One-way link delay (ms).

    Return
         (cubic - aurora) reward
    '''
    assert MODEL_PATH != ""
    cmd = "python evaluate_cubic.py --bandwidth {} --delay {} --queue {} " \
        "--loss {} --duration {}".format(bandwidth, delay, int(queue),
                                         loss, 10)
    print(cmd)
    cubic_reward = float(subprocess.check_output(cmd, shell=True).strip())
    cmd = "CUDA_VISIBLE_DEVICES='' python evaluate_aurora.py --bandwidth {} --delay {} --queue {} "  \
        "--loss {} --model-path {} --duration {}".format(
            bandwidth, delay, int(queue), loss, MODEL_PATH, 10)
    print(cmd)
    aurora_reward = float(subprocess.check_output(cmd, shell=True).strip())
    # print("cubic", cubic_reward, "aurora", aurora_reward)
    return cubic_reward - aurora_reward


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
    params_range = select_range_from_opt_res(optimizer.res)
    return best_param, params_range


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
    global MODEL_PATH
    args = parse_args()
    print(args)
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
    config = read_json_file(args.config_file)[0]
    pbounds = {k: config[k] for k in config if k != 'weight'}
    ckpt_path = initalize_pretrained_model(pbounds, 5, "test_bo", 10, 10000)
    save_dir = args.save_dir
    # Example Flow:
    for i in range(12):
        best_param, params_range = do_bo(
            pbounds, black_box_function, seed=args.seed)
        min_bw, max_bw = params_range['bandwidth']
        min_delay, max_delay = params_range['delay']
        min_loss, max_loss = params_range['loss']
        min_queue, max_queue = params_range['queue']

        print("BO choose paramter range ", params_range)
        rand_ranges = RandomizationRanges(config_file)
        rand_ranges.add_range((min_bw), (min_delay, max_delay),
                              (min_loss, max_loss), (min_queue, max_queue))
        new_config_file = os.path.join("test_bo", "bo_"+str(i) + ".json")
        rand_ranges.dump(new_config_file)
        config_file = new_config_file

        # Use the new param, add more traces into Pensieve, train more based on
        # before
        # latest_ckpt_path, latest_model_path = latest_model_from(save_dir)

        # train Aurora with new parameter
        # command = "python train_rl.py " \
        #     "--total-timesteps {} " \
        #     "--save-dir={save_dir} " \
        #     "--pretrained-model-path {model_path} " \
        #     "--delay {min_delay} {max_delay} " \
        #     "--queue {min_queue} {max_queue} " \
        #     "--bandwidth {min_bandwidth} {max_bandwidth} " \
        #     "--loss {min_loss} {max_loss} " \
        #     "--randomization-range-file {config_file}".format(
        #         720000, save_dir=save_dir,
        #         model_path=latest_ckpt_path,
        #         min_delay=min_delay/1000,
        #         max_delay=max_delay/1000,
        #         min_queue=min_queue,
        #         max_queue=max_queue,
        #         min_bandwidth=min_bandwidth,
        #         max_bandwidth=max_bandwidth,
        #         min_loss=min_loss,
        #         max_loss=max_loss,
        #         config_file=config_file)
        command = "python train_rl.py " \
            "--total-timesteps {} " \
            "--save-dir={save_dir} " \
            "--pretrained-model-path {model_path} " \
            "--randomization-range-file {config_file}".format(
                10000000000, save_dir=save_dir,
                model_path=latest_ckpt_path,
                config_file=config_file)
        # print(command)
        # subprocess.check_call(command.split())
        # latest_ckpt_path, latest_model_path = latest_model_from(save_dir)
        # os.system("rm -r /tmp/best_model_to_serve")
        # print("cp -r {} /tmp/best_model_to_serve".format(latest_model_path))
        # os.system("cp -r {} /tmp/best_model_to_serve".format(latest_model_path))
        #
        # print("Get the file and pass it to the training script, if it exists.\n")
        # print("Running training:", i)


if __name__ == "__main__":
    main()
