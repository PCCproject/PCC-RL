import gym
import network_sim
import tensorflow as tf
from fgm import fgm
import numpy as np

import matplotlib.pyplot as plt

from adversary_wrapper import AdversaryWrapper
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO1
from stable_baselines import TRPO
from simple_arg_parse import arg_or_default

arch_str = arg_or_default("--arch", default="32,16")
if arch_str == "":
    arch = []
else:
    arch = [int(layer_width) for layer_width in arch_str.split(",")]
print("Architecture is: %s" % str(arch))

training_sess = None

class MyMlpPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, net_arch=[{"pi":arch, "vf":arch}],
                                        feature_extraction="mlp", **_kwargs)
        global training_sess
        training_sess = sess

env = gym.make('PccNs-v0')

gamma = arg_or_default("--gamma", default=0.99)
print("gamma = %f" % gamma)
model = PPO1(MyMlpPolicy, env, verbose=1, schedule='constant', timesteps_per_actorbatch=8192, optim_batchsize=2048, gamma=gamma)

#model.learn(total_timesteps=(100 * 410))

with model.graph.as_default():
    saver = tf.train.Saver()
    saver.restore(training_sess, "../mlp/base_10hist_model.ckpt")

#model.learn(total_timesteps=(10 * 410))

obs_ph = model.policy_pi.obs_ph
act = model.policy_pi.deterministic_action

"""

new_env = gym.make('PccNs-v0')
obs = new_env.reset()
rewards = []
for ep in range(0, 30):
    ep_reward = 0
    states = model.policy_pi.initial_state
    for i in range(0, 400):
        #obs = obs.reshape(-1, *obs.shape)
        obs = obs.reshape(1, -1)
        action, vpred, states, _ = model.policy_pi.step(obs, states, False)
        #print("action: %s" % action)
        obs, reward, done, info = new_env.step(action[0])
        ep_reward += reward
    rewards.append(ep_reward)
    obs = new_env.reset()
print("Training reward: %f" % np.mean(rewards))

new_env = gym.make('PccNs-v0')
obs = new_env.reset()
rewards = []
printed = 0
for ep in range(0, 30):
    ep_reward = 0
    states = model.policy_pi.initial_state
    for i in range(0, 400):
        #obs = obs.reshape(-1, *obs.shape)
        obs = obs.reshape(1, -1)
        if printed < 10:
            #print("My obs: %s" % obs)
            printed += 1
        action = training_sess.run(act, feed_dict={obs_ph: obs})
        #print("action: %s" % action)
        obs, reward, done, info = new_env.step(action[0])
        ep_reward += reward
    rewards.append(ep_reward)
    obs = new_env.reset()
print("Testing reward: %f" % np.mean(rewards))

"""

all_eps = [0.00, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.2, 0.3]
model_paths = {"./base_10hist_model.ckpt": "Short Training", "../mlp/base_10hist_model.ckpt": "Full Training"}
model_paths = {"../mlp/base_10hist_model.ckpt": "Full Training"}

fig, axis = plt.subplots()

for model_path, model_name in model_paths.items():
    with model.graph.as_default():
        saver = tf.train.Saver()
        saver.restore(training_sess, model_path)
    means = []
    mins = []
    maxes = []
    for eps in all_eps:
        adv_env = AdversaryWrapper(gym.make('PccNs-v0'), model, training_sess, ord=np.inf, eps=eps)
        obs = adv_env.reset()
        rewards = []
        for ep in range(0, 15):
            ep_reward = 0
            for i in range(0, 400):
                obs = obs.reshape(1, -1)
                action = training_sess.run(act, feed_dict={obs_ph: obs})
                #print("action: %s" % action)
                obs, reward, done, info = adv_env.step(action[0])
                ep_reward += reward
            rewards.append(ep_reward)
            obs = adv_env.reset()
        means.append(np.mean(rewards))
        mins.append(np.min(rewards))
        maxes.append(np.max(rewards))
        print("Average adversarial reward %f: %f" % (eps, np.mean(rewards)))
    axis.semilogx(all_eps, means, label=model_name)
    axis.fill_between(all_eps, mins, maxes, alpha=0.5)
    
plt.savefig("adversary_results.pdf")

print("Obs shape: %s" % obs_ph.shape)
print("Act shape: %s" % act.shape)
with model.graph.as_default():
    error_mult = tf.placeholder(tf.float32)
    adv_inputs, bad_actions = fgm(obs_ph, act, error_mult, ord=1)
#training_sess.run(tf.global_variables_initializer())
good_output = training_sess.run(act, feed_dict={obs_ph: [[0.0, 1.0, 1.0]]})[0][0]
print("True output: %f" % good_output)
this_adv_input, bad_action = training_sess.run([adv_inputs, bad_actions], feed_dict={error_mult: -1.0, obs_ph: [[0.0, 1.0, 1.0]]})
print("Adversarial input %s" % this_adv_input)
print("Bad action: %s" % bad_action)
this_adv_output = training_sess.run(act, feed_dict={obs_ph: this_adv_input})
print("Adversarial output %s" % this_adv_output)
test_inputs = [
[[0.3, 1.0, 1.0]],
[[0.0, 1.3, 1.0]],
[[0.0, 1.0, 1.3]],
[[-0.3, 1.0, 1.0]],
[[0.0, 0.7, 1.0]],
[[0.0, 1.0, 0.7]]
]
best_output = None
best_error = 1e12
farthest_output = None
farthest_error = 0.0
for test_input in test_inputs:
    output = training_sess.run(act, feed_dict={obs_ph: test_input})
    diff = output[0][0] - bad_action
    error = diff * diff
    diff = output[0][0] - good_output
    dist = np.sqrt(diff * diff)
    print("Input %s -> %s, err: %f, dist: %f" % (test_input, output, error, dist))
    if error < best_error:
        best_output = output[0][0]
        best_error = error
    if dist > farthest_error:
        farthest_output = output[0][0]
        farthest_error = dist 

print("Best output: %f, Adv output %f" % (best_output, this_adv_output[0][0]))
print("Farthest output: %f, Adv output %f" % (farthest_output, this_adv_output[0][0]))

exit(-1)

for i in range(0, 6):
    with model.graph.as_default():                                                                   
        saver = tf.train.Saver()                                                                     
        #saver.save(training_sess, "/home/pcc/spec_model_%d.ckpt" % i)
    model.learn(total_timesteps=(1600 * 410))

##
#   Save the model to the location specified below.
##
default_export_dir = "/tmp/pcc_saved_models/model_A/"
export_dir = arg_or_default("--model-dir", default=default_export_dir)
with model.graph.as_default():

    pol = model.policy_pi#act_model

    obs_ph = pol.obs_ph
    act = pol.deterministic_action
    sampled_act = pol.action

    obs_input = tf.saved_model.utils.build_tensor_info(obs_ph)
    outputs_tensor_info = tf.saved_model.utils.build_tensor_info(act)
    stochastic_act_tensor_info = tf.saved_model.utils.build_tensor_info(sampled_act)
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"ob":obs_input},
        outputs={"act":outputs_tensor_info, "stochastic_act":stochastic_act_tensor_info},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    #"""
    signature_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                     signature}

    model_builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    model_builder.add_meta_graph_and_variables(model.sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map=signature_map,
        clear_devices=True)
    model_builder.save(as_text=True)
