import gym
import tensorflow as tf
import numpy as np
from fgm import fgm
import random
import os

class AdversaryWrapper(gym.Env):
    
    def __init__(self, real_env, model, sess, ord=1, eps=0.3):
        self.env = real_env
        self.should_interfere = True
        self.model = model
        self.sess = sess
        self.obs_ph = model.policy_pi.obs_ph
        self.act = model.policy_pi.deterministic_action

        with self.model.graph.as_default():
            self.error_mult = tf.placeholder(tf.float32)
            self.adv_inputs, _ = fgm(self.obs_ph, self.act, self.error_mult, ord=ord, eps=eps)

        self.reversed = 0
        self.total = 0

    def make_adversarial_obs(self, obs):
        adv_obs = self.sess.run(self.adv_inputs, feed_dict={self.error_mult: -10.0,
                self.obs_ph: np.reshape(obs, (1, -1))})
        old_action = self.sess.run(self.act, feed_dict={self.obs_ph: np.reshape(obs, (1, -1))})[0][0]
        adv_action = self.sess.run(self.act, feed_dict={self.obs_ph: adv_obs})[0][0]
        self.total += 1
        if old_action * adv_action < 0:
            self.reversed += 1
        if random.randint(0, 200) < 1:
            #print("Obs diff: %s" % (adv_obs - np.reshape(obs, (1, -1))))
            #print("Action: %s -> %s (%0.1f reversed)" % (old_action, adv_action, 100.0 * self.reversed / self.total))
            print("\t%0.1f reversed" % (100.0 * self.reversed / self.total))
            if random.randint(0, 10) < 1:
                os.system("echo \"%s:%s:%s:%s\nT\" >> ./reversed.txt" % (obs, 
                        old_action, adv_obs, adv_action))
        return adv_obs

    def seed(self, seed=None):
        self.env.seed(seed)

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        if self.should_interfere:
            obs = self.make_adversarial_obs(obs)
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        self.env.render()

    def close(self):
        self.env.close()
