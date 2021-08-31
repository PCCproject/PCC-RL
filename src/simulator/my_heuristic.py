import numpy as np
import ipdb


class MyHeuristic:
    def __init__(self) -> None:
        self.max_tput = 0  # bits/second
        self.min_rtt = 10  # second
        self.start_stage = True
        self.recv_rate_cache = []
        pass

    def step(self, obs, mi):
        recv_rate = round(mi.get("recv rate"), 3)  # bits/sec
        send_rate = round(mi.get("send rate"), 3)  # bits/sec
        avg_latency = mi.get('avg latency')
        lat_diff = mi.rtt_samples[-1] - mi.rtt_samples[0]

        # half = int(len(mi.rtt_samples) / 2)
        # if half >= 1:
        #     lat_diff = np.mean(mi.rtt_samples[half:]) - np.mean(mi.rtt_samples[:half])
        # else:
        #     lat_diff =  0.0
        # lat_diff = mi.rtt_samples[-1] - mi.rtt_samples[0]

        # thresh = 0.002 / self.min_rtt
        # if -1*thresh <= lat_diff / mi.rtt_samples[0] <= thresh:
        #     lat_diff = 0
        self.recv_rate_cache.append(recv_rate)
        if len(self.recv_rate_cache) >6:
            self.recv_rate_cache = self.recv_rate_cache[1:]
        self.min_rtt = min(self.min_rtt, min(mi.rtt_samples))
        self.max_tput = max(self.recv_rate_cache)
        # ipdb.set_trace()
        if self.start_stage and lat_diff == 0 and mi.rtt_samples[-1] / self.min_rtt == 1:  # no latency change and no queue build up so probe bandwidth
            action = 2
        elif lat_diff == 0 and mi.rtt_samples[-1] / self.min_rtt == 1:  # no latency change and no queue build up so probe bandwidth
            # self.max_tput = max(recv_rate, self.max_tput)
            new_send_rate = self.max_tput * 1.25
            action = compute_action(send_rate, new_send_rate)
        elif lat_diff == 0 and mi.rtt_samples[-1] / self.min_rtt != 1:  # no latency change but queue is full so drain queue
            self.start_stage = False
            new_send_rate = max(0.1, self.max_tput - (mi.rtt_samples[-1] - self.min_rtt) * self.max_tput / avg_latency)
            action = compute_action(send_rate, new_send_rate)
        elif lat_diff > 0:  # latency increase
            self.start_stage = False
            # self.max_tput = recv_rate # , self.max_tput)
            new_send_rate = max(0.1, self.max_tput - (mi.rtt_samples[-1] - self.min_rtt) * self.max_tput / avg_latency)
            action = compute_action(send_rate, new_send_rate)

        elif lat_diff < 0 and mi.rtt_samples[-1] / self.min_rtt == 1:  # latency decrease
            self.start_stage = False
            # self.max_tput = max(recv_rate, self.max_tput)
            action = compute_action(send_rate, self.max_tput)
        else:  # latency decrease
            self.start_stage = False
            new_send_rate = max(0.1, self.max_tput - (mi.rtt_samples[-1] - self.min_rtt) * self.max_tput / avg_latency)
            action = compute_action(send_rate, new_send_rate)
        print(lat_diff, action)

        return np.array([action])


def stateless_step(send_rate, avg_latency, lat_diff, start_stage, max_tput, min_rtt, latest_rtt):
    # ipdb.set_trace()
    if lat_diff == 0 and start_stage:  # no latency change
        # new_send_rate = self.max_tput * 1.25
        # action = compute_action(send_rate, new_send_rate)
        action = 2
    elif lat_diff == 0 and not start_stage:  # no latency change
        new_send_rate = max_tput * 1.25
        action = compute_action(send_rate, new_send_rate)
    elif lat_diff > 0:  # latency increase
        new_send_rate = max(0.1, max_tput - (latest_rtt - min_rtt) * max_tput / avg_latency)
        action = compute_action(send_rate, new_send_rate)

    else:  # latency decrease
        action = compute_action(send_rate, max_tput)

    return np.array([action])

def compute_action(send_rate, new_send_rate):
    if new_send_rate > send_rate:
        action = new_send_rate / send_rate - 1
    elif new_send_rate < send_rate:
        action = 1 - send_rate / new_send_rate
    else:
        action = 0
    return action

    # def step(self, obs, mi):
    #     print(obs[-3:].astype(float))
    #     recv_rate = round(mi.get("recv rate"), 3)  # bits/sec
    #     send_rate = round(mi.get("send rate"), 3)  # bits/sec
    #     send_ratio = (send_rate / recv_rate) if recv_rate != 0 else 1
    #     print("send_rate={:.3f}, recv_rate={:.3f}, send_ratio={:.3f}".format(send_rate, recv_rate, send_ratio))
    #     if (float(round(obs[-3], 3))+0) == 0 and round(float(obs[-2]), 5) == 1 and send_ratio > 1:
    #         action = 1
    #     elif (float(round(obs[-3], 3))+0) == 0 and round(float(obs[-2]), 5) == 1:
    #         action = 0.1
    #     elif obs[-2] > 1 and send_ratio > 1:
    #         action = 1 - send_ratio * obs[-2] # mi.rtt_samples[-1] / mi.get('conn min latency')
    #         # return np.array([1 - send_ratio])
    #         # return np.array([1- mi.rtt_samples[-1] / mi.get('conn min latency')])
    #         # return np.array([1 - obs[-2]])
    #     elif obs[-3] > 0 and obs[-2] > 1:
    #         action = 1 - obs[-2]
    #     elif obs[-3] < 0 and obs[-2] > 1 and send_ratio < 1:
    #         action = 1 - send_ratio
    #     else:
    #         action = 0
    #     print("action={}".format(action))
    #     # ipdb.set_trace()
    #
    #     return np.array([action])
