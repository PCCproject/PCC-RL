import csv
import os

import gym
import numpy as np

from plot_scripts.plot_packet_log import PacketLog
from simulator import good_network_sim
from simulator.trace import Trace


class Cubic:
    def __init__(self, seed: int):
        self.seed = seed
        return

    def test(self, trace: Trace, save_dir: str):
        env = gym.make('cubic-v0', traces=[trace], congestion_control_type='cubic',
                       log_dir=save_dir)
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
        with open(os.path.join(save_dir, "cubic_packet_log.csv"), 'w', 1) as f:
            pkt_logger = csv.writer(f, lineterminator='\n')
            pkt_logger.writerows(env.net.pkt_log)
        pkt_log = PacketLog.from_log(env.net.pkt_log)

        return np.mean(rewards), pkt_log.get_reward("", trace)
