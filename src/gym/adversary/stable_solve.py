import gym
import network_sim
import tensorflow as tf
from fgm import fgm
import numpy as np

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

env = gym.make('MountainCarContinuous-v0')
#env = gym.make('PccNs-v0')
#env = gym.make('CartPole-v0')

gamma = arg_or_default("--gamma", default=0.99)
print("gamma = %f" % gamma)
model = PPO1(MyMlpPolicy, env, verbose=1, schedule='constant', timesteps_per_actorbatch=16*8192, optim_batchsize=16*2048, gamma=gamma)

for i in range(0, 6):
    with model.graph.as_default():                                                                   
        saver = tf.train.Saver()                                                                     
        saver.save(training_sess, "./mcc_model_%d.ckpt" % i)
    model.learn(total_timesteps=(16*1000 * 200))
