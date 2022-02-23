from typing import List

import gym
import numpy as np
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding

from common import sender_obs
from simulator.network_simulator.constants import BYTES_PER_PACKET
from simulator.network_simulator.link import Link
from simulator.network_simulator.network import Network
from simulator.network_simulator.pcc.aurora.aurora_sender import AuroraSender
from simulator.network_simulator.pcc.aurora.schedulers import Scheduler


class AuroraEnvironment(gym.Env):

    def __init__(self, trace_scheduler: Scheduler, history_len: int = 10,
                 # features="sent latency inflation,latency ratio,send ratio",
                 features: List[str] = ["sent latency inflation",
                                        "latency ratio", "recv ratio"],
                 record_pkt_log: bool = False):
        """Network environment used in simulation."""
        self.record_pkt_log = record_pkt_log
        self.trace_scheduler = trace_scheduler
        self.current_trace = self.trace_scheduler.get_trace()

        self.history_len = history_len
        self.features = features

        # construct sender and network
        self.links = [Link(self.current_trace), Link(self.current_trace)]
        self.senders = [AuroraSender(
            10 * BYTES_PER_PACKET / (self.current_trace.get_delay(0) * 2/1000),
            self.features, self.history_len, 0, 0, self.current_trace)]
        self.net = Network(self.senders, self.links, self.record_pkt_log)
        self.run_dur = 0.01
        self.steps_taken = 0

        self.action_space = spaces.Box(
            np.array([-1e12]), np.array([1e12]), dtype=np.float32)

        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(
            np.tile(single_obs_min_vec, self.history_len),
            np.tile(single_obs_max_vec, self.history_len), dtype=np.float32)

        self.reward_sum = 0.0
        self.reward_ewma = 0.0

        self.episodes_run = -1

    def seed(self, seed=None):
        self.rand, seed = seeding.np_random(seed)
        return [seed]

    def _get_all_sender_obs(self):
        assert self.senders
        sender_obs = self.senders[0].get_obs()
        sender_obs = np.array(sender_obs).reshape(-1,)
        return sender_obs

    def step(self, action):
        assert self.senders
        self.senders[0].apply_rate_delta(action)
        self.senders[0].on_mi_start()
        self.net.run(self.run_dur)
        # TODO: one run stops and aurora sender should do something
        reward, self.run_dur = self.senders[0].on_mi_finish()
        self.steps_taken += 1
        sender_obs = self._get_all_sender_obs()

        should_stop = self.current_trace.is_finished(self.net.get_cur_time())

        self.reward_sum += reward
        return sender_obs, reward, should_stop, {}

    def reset(self):
        self.steps_taken = 0
        self.net.reset()
        self.current_trace = self.trace_scheduler.get_trace()
        self.current_trace.reset()
        self.run_dur = 0.01
        self.links = [Link(self.current_trace), Link(self.current_trace)]
        self.senders = [AuroraSender(
            10 * BYTES_PER_PACKET / (self.current_trace.get_delay(0) * 2/1000),
            self.features, self.history_len, 0, 0, self.current_trace)]
        self.net = Network(self.senders, self.links, self.record_pkt_log)
        self.episodes_run += 1
        self.senders[0].on_mi_start()
        self.net.run(self.run_dur)
        _, self.run_dur = self.senders[0].on_mi_finish()

        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        self.reward_sum = 0.0
        return self._get_all_sender_obs()


register(id='AuroraEnv-v0', entry_point='simulator.network_simulator.pcc.aurora.aurora_environment:AuroraEnvironment')
