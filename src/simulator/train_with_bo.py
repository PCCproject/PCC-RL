# This is the main training setup for training a Pensieve model, but
# stopping at intervals to add new training data based on performance
# (increases generalizability, we hope!)
import os
import subprocess
import sys

import gym
import numpy as np
from bayes_opt import BayesianOptimization

# from simulator.network_simulator import network
from simulator import good_network_sim as network_sim

# Inputs:
#
# - experiment results directory
# - training data directory (with traces in subdirectories)
# - total epochs
# - bayesian optimizer interval (e.g., every 5000 epochs) - this is how many epoch each training run will go

# Defaults
# Improvement: Probably better if replaced with argparse and passed in (later)
# TOTAL_EPOCHS = 10000
# BAYESIAN_OPTIMIZER_INTERVAL = 1000
TRAINING_DATA_DIR = "../BO-data/randomize-BW-TS/train-bo-added/"
VAL_TRACE_DIR = '../BO-data/randomize-BW-TS/fixed-test-bo'
RESULTS_DIR = "../BO-results/randomize-BW-TS-2-new/"
# NN_MODEL='../new-DR-results/sanity-check-2/model_saved/nn_model_ep_33200.ckpt'

# num_training_runs = int(TOTAL_EPOCHS / BAYESIAN_OPTIMIZER_INTERVAL)


def map_log_to_lin(x):
    x_lin = 2**(10*x)
    return x_lin


def latest_actor_from(path):
    """
    Returns latest tensorflow checkpoint file from a directory.
    Assumes files are named:
    nn_model_ep_<EPOCH#>.ckpt.meta
    """
    def mtime(f): return os.stat(os.path.join(path, f)).st_mtime
    files = list(sorted(os.listdir(path), key=mtime))
    actors = [a for a in files if "nn_model_ep_" in a]
    actor_path = str(path + '/' + actors[-1])
    return os.path.splitext(actor_path)[0]


def black_box_function(x, y):
    '''
    :param x: input is the current params for max-BW
    :param y: input is the current params for TS
    :return: reward is the mpc-rl reward
    '''
    # TODO: this need to be args.summary_dir
    # TODO: do i need to load the actor_path here?
    path = os.path.join(RESULTS_DIR, 'model_saved')
    latest_model_path = latest_actor_from(path)

    x_map = map_log_to_lin(x)

    command = "python rl_test.py  \
                --CURRENT_PARAM_BW={current_max_BW_param} \
                --CURRENT_PARAM_TS={current_max_TS_param} \
                --test_trace_dir='../data/example_traces/' \
                --summary_dir='../MPC_RL_test_results/' \
                --model_path='{model_path}' \
                ".format(current_max_BW_param=x_map, current_max_TS_param=y, model_path=latest_model_path)

    r = float(subprocess.check_output(command, shell=True, text=True).strip())
    return r


def eval_cubic(delay):
    env = gym.make('PccNs-v0', congestion_control_type='cubic',
                   log_dir="/tmp/cubic_log")
    env.seed(42)
    env.set_ranges(1000, 1000, delay, delay, 0, 0, 100, 100)
    _ = env.reset()
    rewards = []
    while True:
        action = [0, 0]
        _, reward, dones, _ = env.step(action)
        rewards.append(reward)
        if dones:
            break
    return np.mean(np.array(rewards))


def eval_aurora(delay):
    env = gym.make('PccNs-v0', log_dir="/tmp/rl_log")
    env.seed(42)
    env.set_ranges(1000, 1000, delay, delay, 0, 0, 100, 100)
    _ = env.reset()
    rewards = []
    while True:
        action = [0, 0]
        _, reward, dones, _ = env.step(action)
        rewards.append(reward)
        if dones:
            break
    return np.mean(np.array(rewards))


def reward_gap_btwn_baseline_aurora(delay):
    '''Compute reward gap between baseline and RL solution.

    Args
        delay: One-way link delay (ms).

    Return
         (cubic - aurora) reward
    '''
    return eval_cubic(delay) - eval_aurora(delay)


def main():
    # Example Flow:
    for i in range(12):
        # if i > 0:
        pbounds = {'delay': (0, 500)}
        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds
            # random_state=2
        )

        optimizer.maximize(
            init_points=13,
            n_iter=2,
            kappa=20,
            xi=0.1
        )
        next = optimizer.max
        delay = next.get('params').get('delay')
        # param_TS = next.get('params').get('y')

        print("BO chose this best param........", delay)

        # Use the new param, add more traces into Pensieve, train more based on
        # before
        path = os.path.join(RESULTS_DIR, 'model_saved')
        latest_model_path = latest_actor_from(path)

        # train Aurora with new parameter

        command = "python multi_agent.py \
                        --TOTAL_EPOCH=5000\
                        --train_trace_dir={training_dir} \
                        --val_trace_dir='{val_dir}'\
                        --summary_dir={results_dir}\
                        --description='first-run' \
                        --nn_model={model_path} \
                        --CURRENT_PARAM_BW={bo_output_param_BW} \
                        --CURRENT_PARAM_TS={bo_output_param_TS} \
                        ".format(training_dir=TRAINING_DATA_DIR, val_dir=VAL_TRACE_DIR,
                                 results_dir=RESULTS_DIR, model_path=latest_model_path,
                                 bo_output_param_BW=bo_best_param_BW, bo_output_param_TS=bo_best_param_TS)

        command = "python multi_agent.py \
                        --TOTAL_EPOCH=5000\
                        --train_trace_dir={training_dir} \
                        --val_trace_dir='{val_dir}'\
                        --summary_dir={results_dir}\
                        --description='first-run' \
                        --nn_model={model_path} \
                        --CURRENT_PARAM_BW={bo_output_param_BW} \
                        --CURRENT_PARAM_TS={bo_output_param_TS} \
                        ".format(training_dir=TRAINING_DATA_DIR, val_dir=VAL_TRACE_DIR,
                                 results_dir=RESULTS_DIR, model_path=latest_model_path,
                                 bo_output_param_BW=bo_best_param_BW, bo_output_param_TS=bo_best_param_TS)
        os.system(command)

        print("Get the file and pass it to the training script, if it exists.\n")
        print("Running training:", i)
        i += 1


if "__name__" == "__main__":
    main()
