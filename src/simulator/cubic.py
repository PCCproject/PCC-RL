import csv
import os

import gym
import numpy as np

from simulator import good_network_sim
from simulator.trace import Trace


class Cubic:
    def __init__(self, save_dir: str, seed: int):
        self.save_dir = save_dir
        self.seed = seed
        return

    def test(self, trace: Trace):
        env = gym.make('cubic-v0', traces=[trace], congestion_control_type='cubic',
                       log_dir=self.save_dir)
        env.seed(self.seed)

        _ = env.reset()
        rewards = []
        while True:
            action = [0, 0]
            _, reward, dones, _ = env.step(action)
            rewards.append(reward * 1000)
            # print(len(rewards), trace.idx, env.net.cur_time)
            if dones:
                break
        with open(os.path.join(self.save_dir, "cubic_packet_log.csv"), 'w', 1) as f:
            pkt_logger = csv.writer(f, lineterminator='\n')
            pkt_logger.writerows(env.net.pkt_log)
        return np.mean(rewards), env.net.pkt_log
