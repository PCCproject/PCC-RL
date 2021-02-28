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
import itertools
import os
import shutil
import time
from typing import List

import gym
import numpy as np
import tensorflow as tf
from stable_baselines import PPO1
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.results_plotter import load_results, ts2xy

from common.utils import read_json_file
# from simulator import network_sim
# from simulator.network_simulator import network
from simulator import good_network_sim

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Training code.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to save the model.")
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma.')
    # parser.add_argument('--arch', type=str, default="32,16", help='arch.')
    parser.add_argument('--config', type=str, required=True,
                        help="UDR config file.")
    parser.add_argument('--range-id', type=int, default=0, help='range id.')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument("--total-timesteps", type=int, default=5000000,
                        help="Total number of steps to be trained.")

    return parser.parse_args()


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, val_envs: List = [], verbose=0, patience=10):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.val_envs = val_envs
        self.val_log_writer = csv.writer(
            open(os.path.join(log_dir, 'validation_log.csv'), 'w', 1),
            lineterminator='\n')
        self.val_log_writer.writerow(
            ['n_calls', 'num_timesteps', 'mean_validation_reward'])
        self.best_val_reward = -np.inf
        self.patience = patience

        self.t_start = time.time()

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
                        saver.save(self.model.sess,
                                   os.path.join(self.log_dir, "pcc_model_best.ckpt"))
                    export_dir = os.path.join(os.path.join(self.log_dir, "best_model_to_serve/"))
                    save_model_to_serve(self.model, export_dir)
            # val_rewards = [test(self.model, val_env)
            #                for val_env in self.val_envs]
            # self.val_log_writer.writerow(
            #     [self.n_calls, self.num_timesteps, np.mean(np.array(val_rewards))])
            # print("{}steps: {}s".format(
            #     self.check_freq, time.time() - self.t_start))
            # self.t_start = time.time()
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
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # print(rewards, dones, info, step_cnt, env.run_dur,
        # env.links[0].print_debug(), env.links[1].print_debug(), env.senders)
        reward_list.append(rewards)
        if dones:
            break
    env.dump_events_to_file(os.path.join(
        env.log_dir, "pcc_env_log_run_{}.json".format(env_id)))
    return reward_list


class MyMlpPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, **_kwargs):
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env,
                                          n_steps, n_batch, reuse, net_arch=[
                                              {"pi": [32, 16], "vf": [32, 16]}],
                                          feature_extraction="mlp", **_kwargs)


def main():
    args = parse_args()
    log_dir = args.save_dir
    os.makedirs(log_dir, exist_ok=True)
    gamma = args.gamma

    env = gym.make('PccNs-v0', log_dir=log_dir)
    env.seed(args.seed)
    env = Monitor(env, log_dir)
    config = read_json_file(args.config)
    print(config)
    env.set_ranges(config["train"]["bandwidth"]["min"],
                   config["train"]["bandwidth"]["max"],
                   config["train"]["latency"]["min"],
                   config["train"]["latency"]["max"],
                   config["train"]["loss"]["min"],
                   config["train"]["loss"]["max"],
                   config["train"]["queue"]["min"],
                   config["train"]["queue"]["max"],)
                   # config["train"]["mss"]["min"],
                   # config["train"]["mss"]["max"])

    bw_list = config["val"]["bandwidth"]
    lat_list = config["val"]["latency"]
    queue_list = config["val"]["queue"]
    loss_list = config["val"]["loss"]
    # mss_list = config["val"]["mss"]
    mss_list = [1500, 1500]
#
    print("gamma = {}" .format(gamma))
    # model = PPO1(MyMlpPolicy, env, verbose=1, schedule='constant',
    #              timesteps_per_actorbatch=8192, optim_batchsize=2048,
    #              gamma=gamma)
    # model = PPO1(MyMlpPolicy, env, verbose=1, seed=args.seed, schedule='constant',
    #              timesteps_per_actorbatch=4000, optim_batchsize=1024,
    #              gamma=gamma)
    model = PPO1(MyMlpPolicy, env, verbose=0, seed=args.seed, schedule='constant',
                 timesteps_per_actorbatch=300, gamma=gamma)

    val_envs = []
    for env_cnt, (bw, lat, loss, queue, mss) in enumerate(
            itertools.product(bw_list, lat_list, loss_list, queue_list, mss_list)):
        # os.makedirs(f'../../results/tmp', exist_ok=True)
        tmp_env = gym.make('PccNs-v0', log_dir=f'../../results/tmp')
        tmp_env.seed(args.seed)
        tmp_env.set_ranges(bw, bw, lat, lat, loss,
                           loss, queue, queue) #, mss, mss)
        val_envs.append(tmp_env)
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=4000, log_dir=log_dir, val_envs=val_envs)

    # print(np.mean([test(model, val_env, env_id) for env_id, val_env in enumerate(val_envs)]))

    # model.learn(total_timesteps=(2 * 1600 * 410), callback=callback)
    # model.learn(total_timesteps=(10000), callback=callback)
    # model.learn(total_timesteps=(100000), callback=callback)
    # model.learn(total_timesteps=(500000), callback=callback)
    # model.learn(total_timesteps=(1000000), callback=callback)
    # model.learn(total_timesteps=(2000000), callback=callback)
    # model.learn(total_timesteps=(80000), callback=callback)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    with model.graph.as_default():
        saver = tf.train.Saver()
        saver.save(model.sess, os.path.join(log_dir, "pcc_model.ckpt"))

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
        model_builder.add_meta_graph_and_variables(model.sess,
                                                   tags=[
                                                       tf.saved_model.tag_constants.SERVING],
                                                   signature_def_map=signature_map,
                                                   clear_devices=True)
        model_builder.save(as_text=True)


if __name__ == '__main__':
    t_start = time.time()
    main()
    print('e2e', time.time() - t_start)
