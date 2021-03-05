import csv
import os
import shutil
import time
import types
from typing import List
import warnings
import logging

import numpy as np
import tensorflow as tf
from udt_plugins.testing.loaded_agent import LoadedModel

if type(tf.contrib) != types.ModuleType:  # if it is LazyLoader
    tf.contrib._warning = None

import gym
from stable_baselines import PPO1
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor

from common.utils import read_json_file, set_tf_loglevel
from simulator import network

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
set_tf_loglevel(logging.FATAL)


class MyMlpPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, **_kwargs):
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env,
                                          n_steps, n_batch, reuse, net_arch=[
                                              {"pi": [32, 16], "vf": [32, 16]}],
                                          feature_extraction="mlp", **_kwargs)


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
        # self.save_path = os.path.join(log_dir, 'saved_models')
        self.save_path = log_dir
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
                            self.save_path, "model_step_{}.ckpt".format(
                                self.n_calls + self.steps_trained)))
                export_dir = os.path.join(os.path.join(
                    self.save_path, "model_to_serve_step_{}/".format(
                        self.n_calls + self.steps_trained)))
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
                val_rewards, loss_list, tput_list, delay_list, send_rate_list = test(
                    self.model, val_env)
                # print(val_env.links[0].print_debug(), "cost {:.3f}".format(time.time() - t_start))
                avg_rewards.append(np.mean(np.array(val_rewards)))
                avg_losses.append(np.mean(np.array(loss_list)))
                avg_tputs.append(float(np.mean(np.array(tput_list))) / 1e6)
                avg_delays.append(np.mean(np.array(delay_list)))
                avg_send_rates.append(
                    float(np.mean(np.array(send_rate_list)))/1e6)
            self.val_log_writer.writerow(
                map(lambda t: "%.3f" % t,
                    [float(self.n_calls), float(self.num_timesteps),
                     np.mean(np.array(avg_rewards)),
                     np.mean(np.array(avg_losses)),
                     np.mean(np.array(avg_tputs)),
                     np.mean(np.array(avg_delays)),
                     np.mean(np.array(avg_send_rates))]))
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

        signature_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                         signature}

        model_builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        model_builder.add_meta_graph_and_variables(
            model.sess, tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_map,
            clear_devices=True)
        model_builder.save(as_text=True)


class Aurora():
    def __init__(self, training_traces, seed, log_dir, timesteps_per_actorbatch,
                 pretrained_model_path=None, gamma=0.99):
        self.seed = seed
        env = gym.make('PccNs-v0', traces=training_traces,
                       log_dir=log_dir, train_flag=True)
        env.seed(seed)
        env = Monitor(env, log_dir)
        self.log_dir = log_dir
        self.steps_trained = 0
        # Load pretrained model
        if pretrained_model_path is not None:
            if pretrained_model_path.endswith('.ckpt'):
                self.model = PPO1(MyMlpPolicy, env, verbose=1, seed=seed,
                                  optim_stepsize=0.001, schedule='constant',
                                  timesteps_per_actorbatch=timesteps_per_actorbatch,
                                  gamma=gamma)
                with self.model.graph.as_default():
                    saver = tf.train.Saver()
                    saver.restore(self.model.sess, pretrained_model_path)
                try:
                    self.steps_trained = int(os.path.splitext(
                        pretrained_model_path)[0].split('_')[-1])
                except:
                    self.steps_trained = 0
            else:
                # model is a tensorflow model to serve
                self.model = LoadedModel(pretrained_model_path)
        else:
            self.model = PPO1(MyMlpPolicy, env, verbose=1, seed=seed,
                              optim_stepsize=0.001, schedule='constant',
                              timesteps_per_actorbatch=timesteps_per_actorbatch,
                              gamma=gamma)
        self.timesteps_per_actorbatch = timesteps_per_actorbatch

    def train(self, validation_traces, total_timesteps):
        assert isinstance(self.model, PPO1)

        val_envs = []
        for _, trace in enumerate(validation_traces):
            tmp_env = gym.make(
                'PccNs-v0', traces=[trace], log_dir=self.log_dir)
            tmp_env.seed(self.seed)
            val_envs.append(tmp_env)

        # Create the callback: check every n steps and save best model
        callback = SaveOnBestTrainingRewardCallback(
            check_freq=self.timesteps_per_actorbatch, log_dir=self.log_dir,
            steps_trained=self.steps_trained, val_envs=val_envs)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def test(self, traces):
        results = []
        for _, trace in enumerate(traces):
            tmp_env = gym.make(
                'PccNs-v0', traces=[trace], log_dir=self.log_dir)
            tmp_env.seed(self.seed)

            reward_list, loss_list, tput_list, delay_list, send_rate_list = test(self.model, tmp_env)
            result = list(zip(reward_list, loss_list, tput_list, delay_list, send_rate_list))
            # envs.append(tmp_env)
            results.append(result)
        return results

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError


def test(model, env, env_id=0):
    reward_list = []
    loss_list = []
    tput_list = []
    delay_list = []
    send_rate_list = []
    obs = env.reset()
    while True:
        if isinstance(model, LoadedModel):
            obs = obs.reshape(1, -1)
            action = model.act(obs)
            action = action['act'][0]
        else:
            action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward_list.append(rewards)
        last_event = env.event_record['Events'][-1]
        loss_list.append(last_event['Loss Rate'])
        delay_list.append(last_event['Latency'])
        tput_list.append(last_event['Throughput'])
        send_rate_list.append(last_event['Send Rate'])
        if dones:
            break
    # env.dump_events_to_file(os.path.join(
    #     env.log_dir, "pcc_env_log_run_{}.json".format(env_id)))
    return reward_list, loss_list, tput_list, delay_list, send_rate_list
