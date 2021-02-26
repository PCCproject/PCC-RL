import argparse
from common import sender_obs
import csv
import itertools
import os
import sys
import time
import shutil

import gym
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
if type(tf.contrib) != type(tf):
    tf.contrib._warning = None
from stable_baselines import PPO1
from stable_baselines.common.policies import FeedForwardPolicy, MlpPolicy

from common.utils import read_json_file
# from simulator import network_sim
from simulator.network_simulator import network as network_sim
from udt_plugins.testing.loaded_agent import LoadedModel

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
import logging, os

logging.disable(logging.WARNING)

def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Pcc Aurora Testing in simulator.")
    parser.add_argument('--model-path', type=str, required=True,
                        help="path to tensorflow model")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to testing results.")
    parser.add_argument('--config', type=str, required=True,
                        help="path to config file in json format.")
    # parser.add_argument('--arch', type=str, default="32,16", help='arch.')
    parser.add_argument('--seed', type=int, default=42, help='seed')

    return parser.parse_args()

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

class MyMlpPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, **_kwargs):
        arch = [32, 16]
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env,
                                          n_steps, n_batch, reuse, net_arch=[
                                              {"pi": arch, "vf": arch}],
                                          feature_extraction="mlp", **_kwargs)


args = parse_args()

env = gym.make('PccNs-v0', log_dir='')

if args.model_path.endswith('.ckpt'):
    # model is a tensorflow checkpoint
    model = PPO1(MyMlpPolicy, env, verbose=1, schedule='constant',
                 timesteps_per_actorbatch=8192, optim_batchsize=2048,
                 gamma=0.99)
    with model.graph.as_default():
        saver = tf.train.Saver()  # save neural net parameters
        nn_model = os.path.join(args.model_path)
        saver.restore(model.sess, nn_model)
else:
    # model is a tensorflow model to serve
    model = LoadedModel(args.model_path)

export_dir = os.path.join(os.path.join(os.path.dirname(args.model_path), "model_to_serve/"))
os.makedirs(export_dir, exist_ok=True)
print(export_dir)
save_model_to_serve(model, export_dir)

sys.exit()

#
# bw_list = [1, 5, 10, 100, 500, 1000, 2000, 5000, 8000]
# bw_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# bw_list = [50, 80, 100, 300, 500, 800, 1000, 1500, 2000, 3000, 4000, 5000]
# bw_list = [100]
config = read_json_file(args.config)
bw_list = config['bandwidth']
lat_list = config['latency']
queue_list = config['queue']
loss_list = config['loss']
# bw_list = [1000]
# bw_list = [50]
# lat_list = [0.05]
# queue_list = [5]
# queue_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# loss_list = [0]

param_sets = itertools.product(bw_list, lat_list, loss_list, queue_list)
param_set_len = len(list(param_sets))
param_sets = itertools.product(bw_list, lat_list, loss_list, queue_list)

for env_cnt, (bw, lat, loss, queue) in enumerate(param_sets):
    # print(bw, lat, loss, queue)
    env.set_ranges(bw, bw, lat, lat, loss, loss, queue, queue)
    obs, valid = env.reset()
    step_cnt = 0
    t_start = time.time()
    # writer = csv.writer(open('obs.csv', "w", 1), delimiter='\t', lineterminator='\n')
    writer = open('obs.csv', "w", 1)
    header = ["action"]
    # for i in range(10):
    #     header += [f"sent latency inflation{i}", f"latency ratio{i}", f"send ratio{i}"]
    # writer.writerow(header)
    writer.write("rate delta\t\tsending rate\t\t[bytes_sent, "
                 "bytes_acked, bytes_lost, send_start, send_end, recv_start, "
                 "recv_end, rtt_samples, packet_size]\n")
    while True:
        # if env.senders[0].got_data:
        if valid:
            if isinstance(model, LoadedModel):
                action = model.act(obs.reshape(1, -1))
                action = action['act'][0]
            else:
                action, _states = model.predict(obs)
        else:
            action = [0]
        # print(obs, action)
        # print(','.join([str(action[0])] + [str(x) for x in list(obs)])+ '\n')
        # writer.writerow(["{:.4f}".format(action[0])] + ["{:.4f}".format(val) for val in obs])
        input_array_line = "["
        for val in env.senders[0].history.as_array()[::-1]:
            if val < 0:
                input_array_line += "{:.3f}".format(val) + ", "
            else:
                input_array_line += "{:.4f}".format(val) + ", "
        input_array_line = input_array_line[:-1] + "]"
        sample = env.senders[0].history.values[-1]
        # sample = env.senders[0].mi_cache[step_cnt]
        # history.values[-1]
        if sample.rtt_samples:
            rtt_samples = [sample.rtt_samples[0], sample.rtt_samples[-1]]
        else:
            rtt_samples = []
        # sample_vector = "[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {}, {:.4f}]".format(sample.bytes_sent, sample.bytes_acked,
        #          sample.bytes_lost,
        #          sample.send_start,
        #          sample.send_end,
        #          sample.recv_start,
        #          sample.recv_end,
        #          rtt_samples,
        sample_vector = "[{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, " \
                "[rtt min={:.4}, mean_rtt={:.4}, 1half rtt mean={:.4}, 2half rtt mean={:.4}, " \
               "latency inc={:.4f}, send dur={:.4f}, recv dur={:.4f}, "\
               "send rate={}, recv rate={:.4f}], {}]".format(-1, sample.bytes_sent, sample.bytes_acked,
                sample.bytes_lost,
                sample.send_start,
                sample.send_end,
                sample.recv_start,
                sample.recv_end,
                float(min(sample.rtt_samples) if sample.rtt_samples else 0),
                float(np.mean(sample.rtt_samples) if sample.rtt_samples else 0),
                float(sample.rtt_samples[0] if sample.rtt_samples else 0),
                float(sample.rtt_samples[1] if sample.rtt_samples else 0),
                float(sample.rtt_samples[1] - sample.rtt_samples[0] if sample.rtt_samples else 0),
                # np.mean(sample.rtt_samples[:int(len(sample.rtt_samples))]),
                # np.mean(sample.rtt_samples[int(len(sample.rtt_samples)):]),
                float(sample.send_end - sample.send_start),
                float(sample.recv_end - sample.recv_start),
                float(sender_obs._mi_metric_send_rate(sample)),
                float(sender_obs._mi_metric_recv_rate(sample)),
                float(sample.packet_size))
        # sample_vector = "[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {}, {:.4f}]".format(sample.bytes_sent, sample.bytes_acked,
        #          sample.bytes_lost,
        #          sample.first_packet_send_time,
        #          sample.last_packet_send_time,
        #          sample.first_packet_ack_time,
        #          sample.last_packet_ack_time,
        #          rtt_samples,
        #          sample.packet_size)


        obs, rewards, dones, info, valid = env.step(action)
        writer.write("{:.4f}\t\t\t{:.4f}\t\t\t{}\t\t\t{}\n".format(action[0], env.senders[0].rate*1500 * 8/1e6, sample_vector, input_array_line))
        # writer.write("{:.4f}\t\t\t{:.4f}".format(action[0], self.senders[0].rate) + ["{:.4f}".format(val) for val in obs])
        step_cnt += 1
        if dones:
            os.makedirs(os.path.join(args.save_dir, "rl_test"), exist_ok=True)
            env.dump_events_to_file(
                os.path.join(args.save_dir, "rl_test", "rl_test_log{}.json".format(env_cnt)))
            print("{}/{}, {}s".format(env_cnt, param_set_len, time.time() - t_start))
            t_start = time.time()
            break
