# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import csv
import logging
import os
import shutil
import time
import types
import warnings
from typing import List
import itertools
import ipdb

import gym
import numpy as np
import tensorflow as tf

if type(tf.contrib) != types.ModuleType:  # if it is LazyLoader
    tf.contrib._warning = None
from common.utils import read_json_file, set_tf_loglevel
from stable_baselines import PPO1
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.results_plotter import load_results, ts2xy

# from simulator import network_sim
# from simulator.network_simulator import network
from simulator import good_network_sim

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore")

set_tf_loglevel(logging.FATAL)


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Training code.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to save the model.")
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma.')
    parser.add_argument("--delay", type=float,  nargs=2, required=True)
    parser.add_argument("--bandwidth", type=float, nargs=2, required=True)
    parser.add_argument("--loss", type=float, nargs=2, required=True)
    parser.add_argument("--queue", type=float, nargs=2, required=True)
    # parser.add_argument('--arch', type=str, default="32,16", help='arch.')
    parser.add_argument('--seed', type=int, default=20, help='seed')
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="Total number of steps to be trained.")
    parser.add_argument("--pretrained-model-path", type=str, default=None,
                        help="Path to a pretrained Tensorflow checkpoint!")
    parser.add_argument("--val-delay", type=float, nargs="+", required=True)
    parser.add_argument("--val-bandwidth", type=float, nargs="+", required=True)
    parser.add_argument("--val-loss", type=float, nargs="+", required=True)
    parser.add_argument("--val-queue", type=float, nargs="+", required=True)

    return parser.parse_args()


def check_args(args):
    """Check arg validity."""
    assert args.delay[0] <= args.delay[1]
    assert args.bandwidth[0] <= args.bandwidth[1]
    assert args.loss[0] <= args.loss[1]
    assert args.queue[0] <= args.queue[1]
    assert args.pretrained_model_path.endswith(".ckpt")


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using
    ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, val_envs: List = [],
                 verbose=0, patience=10, steps_trained=0):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.val_envs = val_envs
        self.val_log_writer = csv.writer(
            open(os.path.join(log_dir, 'validation_log.csv'), 'w', 1),
            delimiter='\t', lineterminator='\n')
        self.val_log_writer.writerow(
            ['n_calls', 'num_timesteps', 'mean_validation_reward', 'loss',
             'throughput', 'latency', 'sending_rate'])
        self.best_val_reward = -np.inf
        self.patience = patience

        self.t_start = time.time()
        self.steps_trained = steps_trained

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                        self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    # self.model.save(self.save_path)
                with self.model.graph.as_default():
                    saver = tf.train.Saver()
                    saver.save(
                        self.model.sess, os.path.join(
                            self.log_dir,
                            "model_step_{}.ckpt".format(self.n_calls + self.steps_trained)))
                export_dir = os.path.join(os.path.join(
                    self.log_dir,
                    "model_to_serve_step_{}/".format(self.n_calls + self.steps_trained)))
                save_model_to_serve(self.model, export_dir)
                # val_rewards = [test(self.model, val_env)
                #                for val_env in self.val_envs]
            avg_rewards = []
            avg_losses = []
            avg_tputs = []
            avg_delays = []
            avg_send_rates = []
            for idx, val_env in enumerate(self.val_envs):
                # print("{}/{} start".format(idx +1, len(self.val_envs)) )
                # t_start = time.time()
                val_rewards, loss_list, tput_list, delay_list, send_rate_list = test(self.model, val_env)
                # print(val_env.links[0].print_debug(), "cost {:.3f}".format(time.time() - t_start))
                avg_rewards.append(np.mean(np.array(val_rewards)))
                avg_losses.append(np.mean(np.array(loss_list)))
                avg_tputs.append(float(np.mean(np.array(tput_list))) / 1e6)
                avg_delays.append(np.mean(np.array(delay_list)))
                avg_send_rates.append(float(np.mean(np.array(send_rate_list)))/1e6)
            self.val_log_writer.writerow(
                map(lambda t: "%.3f" % t, [float(self.n_calls), float(self.num_timesteps), np.mean(np.array(avg_rewards)),
                 np.mean(np.array(avg_losses)), np.mean(np.array(avg_tputs)),
                 np.mean(np.array(avg_delays)), np.mean(np.array(avg_send_rates))]))
            print("val every{}steps: {}s".format(
                self.check_freq, time.time() - self.t_start))
            self.t_start = time.time()
            # # if self.patience == 0:
            # #     return False
            # if np.mean(np.array(val_rewards)) > self.best_val_reward:  # type: ignore
            #     self.best_val_reward = np.mean(np.array(val_rewards))
            #     self.patience = 10
            #     print('val improved')
            # else:
            #     self.patience -= 1
            #     print('val no improvement, patience {}'.format(self.patience))
        return True


def test(model, env, env_id=0):
    reward_list = []
    loss_list = []
    tput_list = []
    delay_list = []
    send_rate_list = []
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # print(rewards, dones, info, step_cnt, env.run_dur,
        # env.links[0].print_debug(), env.links[1].print_debug(), env.senders)
        reward_list.append(rewards)
        last_event = env.event_record['Events'][-1]
        loss_list.append(last_event['Loss Rate'])
        delay_list.append(last_event['Latency'])
        tput_list.append(last_event['Throughput'])
        send_rate_list.append(last_event['Send Rate'])
        # print("last event", last_event['Send Rate']/1500/8)
        if dones:
            break
    env.dump_events_to_file(os.path.join(
        env.log_dir, "pcc_env_log_run_{}.json".format(env_id)))
    return reward_list, loss_list, tput_list, delay_list, send_rate_list


class MyMlpPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, **_kwargs):
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env,
                                          n_steps, n_batch, reuse, net_arch=[
                                              {"pi": [32, 16], "vf": [32, 16]}],
                                          feature_extraction="mlp", **_kwargs)


def main():
    args = parse_args()
    min_delay, max_delay = args.delay
    min_loss, max_loss = args.loss
    min_queue, max_queue = args.queue
    min_bandwidth, max_bandwidth = args.bandwidth
    log_dir = args.save_dir
    os.makedirs(log_dir, exist_ok=True)
    gamma = args.gamma

    env = gym.make('PccNs-v0', log_dir=log_dir, max_steps=200, train_flag=True)
    env.seed(args.seed)
    env = Monitor(env, log_dir)
    # config = read_json_file(args.config)
    # print(config)
    env.set_ranges(min_bandwidth, max_bandwidth,
                   min_delay, max_delay,
                   min_loss, max_loss,
                   min_queue, max_queue)
    # config["train"]["mss"]["min"],
    # config["train"]["mss"]["max"])
#
#     bw_list = config["val"]["bandwidth"]
#     lat_list = config["val"]["latency"]
#     queue_list = config["val"]["queue"]
#     loss_list = config["val"]["loss"]
#     mss_list = config["val"]["mss"]

    print("gamma = {}" .format(gamma))
    # model = PPO1(MyMlpPolicy, env, verbose=1, schedule='constant',
    #              timesteps_per_actorbatch=8192, optim_batchsize=2048,
    #              gamma=gamma)
    # model = PPO1(MyMlpPolicy, env, verbose=1, seed=args.seed, schedule='constant',
    #              timesteps_per_actorbatch=4000, optim_batchsize=1024,
    #              gamma=gamma)

    # Initialize model and agent policy
    model = PPO1(MyMlpPolicy, env, verbose=0, seed=args.seed, optim_stepsize=0.001,
                 schedule='constant', timesteps_per_actorbatch=7200, gamma=gamma)

    steps_trained = 0
    # Load pretrained model
    if args.pretrained_model_path is not None and args.pretrained_model_path:
        with model.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(model.sess, args.pretrained_model_path)
        try:
            steps_trained = int(os.path.splitext(
                args.pretrained_model_path)[0].split('_')[-1])
        except:
            steps_trained = 0

    val_envs = []
    for env_cnt, (bw, lat, loss, queue) in enumerate(
            itertools.product(args.val_bandwidth, args.val_delay,
                args.val_loss, args.val_queue)):
        # os.makedirs(f'../../results/tmp', exist_ok=True)
        tmp_env = gym.make('PccNs-v0', log_dir=log_dir, max_steps=200)
        tmp_env.seed(args.seed)
        tmp_env.set_ranges(bw, bw, lat, lat, loss, loss, queue, queue)
        val_envs.append(tmp_env)

    # Create the callback: check every n steps and save best model
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=7200, log_dir=log_dir, steps_trained=steps_trained, val_envs=val_envs)

    # model.learn(total_timesteps=(2 * 1600 * 410), callback=callback)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    with model.graph.as_default():
        saver = tf.train.Saver()
        saver.save(model.sess, os.path.join(log_dir, "model_to_serve.ckpt"))

    # Save the model to the location specified below.
    export_dir = os.path.join(os.path.join(log_dir, "model_to_serve/"))
    save_model_to_serve(model, export_dir)


def save_model_to_serve(model, export_dir):
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    with model.graph.as_default():

        pol = model.policy_pi  # act_model

        obs_ph = pol.obs_ph
        act = pol.deterministic_action
        sampled_act = pol.action

        obs_input = tf.saved_model.utils.build_tensor_info(obs_ph)
        outputs_tensor_info = tf.saved_model.utils.build_tensor_info(act)
        stochastic_act_tensor_info = tf.saved_model.utils.build_tensor_info(
            sampled_act)
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"ob": obs_input},
            outputs={"act": outputs_tensor_info,
                     "stochastic_act": stochastic_act_tensor_info},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        # """
        signature_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                         signature}

        model_builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        model_builder.add_meta_graph_and_variables(
            model.sess, tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_map,
            clear_devices=True)
        model_builder.save(as_text=True)


if __name__ == '__main__':
    t_start = time.time()
    main()
    print('e2e', time.time() - t_start)
