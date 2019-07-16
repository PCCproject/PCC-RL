#  python3 make_graph.py --num-agents 2 --dump-rate 1000 --save-rate 100 --in-out 2 2 --log-range 1 -1 --criteria all

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--num-agents", type=int, default=1, help="number of good agents")
    parser.add_argument("--save-rate", type=int, default=100, help="dump rate (iteration) of json files")
    parser.add_argument("--dump-rate", type=int, default=1000, help="dump rate (iteration) of json files")
    parser.add_argument("--in-out", type=str, default=[0, 0], help="[index of input, index of output]", nargs=2)
    parser.add_argument("--log-range", type=int, default=[1,-1], help="range of log files to plot", nargs=2)
    parser.add_argument("--criteria", type=str, default="reward", help="what attributes to plot: reward, send-rate, throughput, latency, loss-rate, all") # reward
    return parser.parse_args()


def plot_avg_criteria(arglist):

    dir = os.getcwd() + "/dump-events" + arglist.in_out[0] + "/"

    start_log = arglist.log_range[0]
    end_log = arglist.log_range[1]

    # one for each agent
    avg_agent_criterias = [[] for _ in range(arglist.num_agents)]
    if (end_log == -1):
        end_log = len(os.listdir(dir))

    criteria = ""
    if(arglist.criteria == "reward"):
        criteria = "Reward"
    elif(arglist.criteria == "send-rate"):
        criteria = "Send Rate"
    elif(arglist.criteria == "throughput"):
        criteria = "Throughput"
    elif(arglist.criteria == "latency"):
        criteria = "Latency"
    elif(arglist.criteria == "loss-rate"):
        criteria = "Loss Rate"

    for i in range(start_log, end_log+1):
        with open(dir + "pcc_env_log_run_%d.json" % (i * arglist.dump_rate)) as f:
            data = json.load(f)
            for ag in range(arglist.num_agents):
                # avg_agent_criterias[ag].append(np.mean([float(rec[criteria]) for rec in data[ag]]))
                for rec in data[ag]:
                    avg_agent_criterias[ag].append(float(rec[criteria]))

    x_range = range((start_log-1)*arglist.dump_rate+arglist.save_rate, (end_log)*arglist.dump_rate+arglist.save_rate, arglist.save_rate)
    for ag in range(arglist.num_agents):
        plt.plot(x_range, avg_agent_criterias[ag], linestyle="-", linewidth=0.5, label='agent {}'.format(ag+1))
    plt.legend()
    plt.savefig("plot_avg_{}_{}.png".format(arglist.criteria, arglist.in_out[1]))
    plt.close()

if __name__ == '__main__':
    arglist = parse_args()
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
