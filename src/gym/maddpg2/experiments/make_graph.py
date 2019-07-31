#  python3 make_graph.py --num-agents 2 --dump-rate 1000 --save-rate 100 --in-out 2 2 --log-range 1 -1 --criteria all

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

title_list = ["reward", "send-rate", "throughput", "latency", "loss-rate"]

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--num-agents", type=int, default=1, help="number of good agents")
    parser.add_argument("--save-rate", type=int, default=100, help="dump rate (iteration) of json files")
    parser.add_argument("--dump-rate", type=int, default=1000, help="dump rate (iteration) of json files")
    parser.add_argument("--in-out", type=str, default=[0, 0], help="[index of input, index of output]", nargs=2)
    parser.add_argument("--log-range", type=int, default=[1,-1], help="range of log files to plot", nargs=2)
    parser.add_argument("--type", type=str, default="epi", help="plot epi or step")
    return parser.parse_args()


def plot_avg_criteria(arglist):

    dir = os.getcwd() + "/dump-events" + arglist.in_out[0] + "/"

    start_log = arglist.log_range[0]
    end_log = arglist.log_range[1]

    # one for each agent
    avg_agent_reward = [[] for _ in range(arglist.num_agents)]
    avg_agent_sendrate = [[] for _ in range(arglist.num_agents)]
    avg_agent_throughput = [[] for _ in range(arglist.num_agents)]
    avg_agent_latency = [[] for _ in range(arglist.num_agents)]
    avg_agent_lossrate = [[] for _ in range(arglist.num_agents)]

    y_list = [avg_agent_reward, avg_agent_sendrate, avg_agent_throughput, avg_agent_latency, avg_agent_lossrate]


    if (end_log == -1):
        end_log = len(os.listdir(dir))

    for i in range(start_log, end_log+1):
        with open(dir + "pcc_env_log_run_%d.json" % (i * arglist.dump_rate)) as f:
            data = json.load(f)
            for ag in range(arglist.num_agents):
                for rec in data[ag]:
                    avg_agent_reward[ag].append(float(rec["Reward"]))
                    avg_agent_sendrate[ag].append(float(rec["Send Rate"]))
                    avg_agent_throughput[ag].append(float(rec["Throughput"]))
                    avg_agent_latency[ag].append(float(rec["Latency"]))
                    avg_agent_lossrate[ag].append(float(rec["Loss Rate"]))

    x_range = range((start_log-1)*arglist.dump_rate+arglist.save_rate, (end_log)*arglist.dump_rate+arglist.save_rate, arglist.save_rate)

    fig, ax = plt.subplots(5,1,figsize=(15,15))
    # fig = plt.figure()
    fig.suptitle("experiment_{}".format(arglist.in_out[1]))
    for idx in range(0, 5):
        ax = plt.subplot(5, 1, idx+1)
        for ag in range(arglist.num_agents):
            ax.plot(x_range, y_list[idx][ag], linestyle="-", linewidth=0.5, label='agent {}'.format(ag+1))
        ax.legend()
        ax.set_title(title_list[idx])

    fig.savefig("plot_experiment_{}.png".format(arglist.in_out[1]))

def plot_steps(arglist):
    dir = os.getcwd() + "/dump-events" + arglist.in_out[0] + "/"

    start_log = arglist.log_range[0]

    # one for each agent
    avg_agent_reward = [[] for _ in range(arglist.num_agents)]
    avg_agent_sendrate = [[] for _ in range(arglist.num_agents)]
    avg_agent_throughput = [[] for _ in range(arglist.num_agents)]
    avg_agent_latency = [[] for _ in range(arglist.num_agents)]
    avg_agent_lossrate = [[] for _ in range(arglist.num_agents)]

    y_list = [avg_agent_reward, avg_agent_sendrate, avg_agent_throughput, avg_agent_latency, avg_agent_lossrate]

    with open(dir + "steps_at_epi_%d.json" % (start_log * 1000)) as f:
        data = json.load(f)
        for ag in range(arglist.num_agents):
            for rec in data[ag]:
                avg_agent_reward[ag].append(float(rec["Reward"]))
                avg_agent_sendrate[ag].append(float(rec["Send Rate"]))
                avg_agent_throughput[ag].append(float(rec["Throughput"]))
                avg_agent_latency[ag].append(float(rec["Latency"]))
                avg_agent_lossrate[ag].append(float(rec["Loss Rate"]))

    x_range = range(1, 101)

    fig, ax = plt.subplots(5,1,figsize=(15,15))
    # fig = plt.figure()
    fig.suptitle("steps_at_api_{}_experiment_{}".format(start_log * 1000, arglist.in_out[1]))
    for idx in range(0, 5):
        ax = plt.subplot(5, 1, idx+1)
        for ag in range(arglist.num_agents):
            ax.plot(x_range, y_list[idx][ag], linestyle="-", linewidth=0.5, label='agent {}'.format(ag+1))
        ax.legend()
        ax.set_title(title_list[idx])

    fig.savefig("steps_at_epi_{}_experiment_{}.png".format(start_log * 1000, arglist.in_out[1]))


if __name__ == '__main__':
    arglist = parse_args()
    if(arglist.type == "epi"):
        plot_avg_criteria(arglist)
    elif(arglist.type == "step"):
        plot_steps(arglist)

    '''
    if (arglist.criteria != "all"):
        plot_avg_criteria(arglist)
    else:
        arglist.criteria = "reward"
        plot_avg_criteria(arglist)
        arglist.criteria = "send-rate"
        plot_avg_criteria(arglist)
        arglist.criteria = "throughput"
        plot_avg_criteria(arglist)
        arglist.criteria = "latency"
        plot_avg_criteria(arglist)
        arglist.criteria = "loss-rate"
        plot_avg_criteria(arglist)
        '''
