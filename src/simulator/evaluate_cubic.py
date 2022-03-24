import matplotlib.pyplot as plt
from simulator.network_simulator.bbr_old import BBR_old
from simulator.network_simulator.cubic import Cubic
from simulator.pantheon_dataset import PantheonDataset
from simulator.pantheon_trace_parser.connection import Connection

# import argparse
# import csv
# import multiprocessing as mp
import os
# import time
# import warnings
#
# import gym
# import numpy as np
# from common.utils import set_seed
#
# from simulator import good_network_sim as network_sim
# from simulator.trace import generate_trace, generate_traces, Trace
#
# warnings.filterwarnings("ignore")
#
#
# def parse_args():
#     """Parse arguments from the command line."""
#     parser = argparse.ArgumentParser("Cubic Testing in simulator.")
#     parser.add_argument('--save-dir', type=str, default="",
#                         help="direcotry to testing results.")
#     parser.add_argument('--delay', type=float, default=50,
#                         help="one-way delay. Unit: millisecond.")
#     parser.add_argument('--bandwidth', type=float, default=2,
#                         help="Constant bandwidth. Unit: mbps.")
#     parser.add_argument('--loss', type=float, default=0,
#                         help="Constant random loss of uplink.")
#     parser.add_argument('--queue', type=int, default=10,
#                         help="Uplink queue size. Unit: packets.")
#     parser.add_argument('--seed', type=int, default=42, help='seed')
#     parser.add_argument('--duration', type=int, default=30,
#                         help='Flow duration in seconds.')
#     parser.add_argument("--config-file", type=str, default=None,
#                         help='config file.')
#     parser.add_argument('--time-variant-bw', action='store_true',
#                         help='Generate time variant bandwidth if specified.')
#     parser.add_argument('--trace-file', type=str, default=None,
#                         help='path to a trace file.')
#
#     return parser.parse_args()
#
#
# def test_on_trace(trace, save_dir, seed):
#     env = gym.make('cubic-v0', traces=[trace], congestion_control_type='cubic',
#                    log_dir=save_dir)
#     env.seed(seed)
#
#     _ = env.reset()
#     rewards = []
#     while True:
#         action = [0, 0]
#         _, reward, dones, _ = env.step(action)
#         rewards.append(reward * 1000)
#         if dones:
#             break
#     with open(os.path.join(save_dir, "cubic_packet_log.csv"), 'w', 1) as f:
#         pkt_logger = csv.writer(f, lineterminator='\n')
#         pkt_logger.writerows(env.net.pkt_log)
#     return rewards, env.net.pkt_log
#
#
# def test_on_traces(traces, save_dirs, seed):
#     n_proc=mp.cpu_count()//2
#     arguments = [(trace, save_dir, seed) for trace, save_dir in zip(traces, save_dirs)]
#     with mp.Pool(processes=n_proc) as pool:
#         results = pool.starmap(test_on_trace, arguments)
#     rewards = [result[0] for result in results]
#     pkt_logs = [result[1] for result in results]
#     # rewards = []
#     # pkt_logs = []
#     # for trace, save_dir in zip(traces, save_dirs):
#     #     reward, pkt_log = test_on_trace(trace, save_dir, seed)
#     #     rewards.append(reward)
#     #     pkt_logs.append(pkt_log)
#     return rewards, pkt_logs
#
#

def plot_pantheon_traces():
    conn_type = 'cellular'
    dataset = PantheonDataset('../../data', conn_type)
    logs = [os.path.join('../../data', conn_type, link_name,
                              trace_name+'.log') for link_name, trace_name in dataset.trace_names]
    for log, (link_name, trace_name) in zip(logs, dataset.trace_names):
        conn = Connection(log, use_cache=False)
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(conn.throughput_timestamps, conn.throughput)
        axes[0].set_xlabel('Time(s)')
        axes[0].set_ylabel('Mbps')
        axes[1].plot(conn.datalink_delay_timestamps, conn.datalink_delay, label='datalink')
        axes[1].plot(conn.acklink_delay_timestamps, conn.acklink_delay, label='acklink')
        axes[1].set_xlabel('Time(s)')
        axes[1].set_ylabel('Latency(ms)')
        axes[1].legend()
        fig.set_tight_layout(True)
        os.makedirs('/datamirror/zxxia/yihua/traces/{}/{}'.format(conn_type, link_name,), exist_ok=True)
        fig.savefig(os.path.join('/datamirror/zxxia/yihua/traces/{}/{}/{}.jpg'.format(conn_type, link_name, trace_name)))
        plt.close()
    # print(save_dirs)
    # pass

save_root = '/datamirror/zxxia/yihua'
def main():
    cubic = Cubic(True)
    bbr_old = BBR_old(True)

    for loss in [0, 0.01]:

        for conn_type in ['ethernet', 'cellular']:
            dataset = PantheonDataset('../../data', conn_type, target_ccs=['bbr', 'vegas'])
            traces = dataset.get_traces(loss, 50)
            save_dirs = [os.path.join(save_root, str(loss), conn_type, link_name,
                                      trace_name) for link_name, trace_name in dataset.trace_names]

            # cellular_dataset = PantheonDataset('../../data', 'cellular', target_ccs=['bbr', 'vegas'])
            # ethernet_dataset.get_traces(0, 100)

            cubic.test_on_traces(traces, [os.path.join(save_dir, cubic.cc_name) for save_dir in save_dirs], True, 8)
            bbr_old.test_on_traces(traces, [os.path.join(save_dir, bbr_old.cc_name) for save_dir in save_dirs], True, 8)

    # args = parse_args()
    # set_seed(args.seed)
    # if args.save_dir:
    #     os.makedirs(args.save_dir, exist_ok=True)
    #
    # if args.trace_file and args.trace_file.endswith('.json'):
    #     test_traces = [Trace.load_from_file(args.trace_file)]
    # elif args.trace_file and args.trace_file.endswith('.log'):
    #     test_traces = [Trace.load_from_pantheon_file(
    #         args.trace_file, args.delay, args.loss, args.queue)]
    #
    # elif args.config_file is not None:
    #     test_traces = generate_traces(args.config_file, 1, args.duration,
    #                                   constant_bw=not args.time_variant_bw)
    # else:
    #     test_traces = [generate_trace((args.duration, args.duration),
    #                                   (args.bandwidth, args.bandwidth),
    #                                   (args.delay, args.delay),
    #                                   (args.loss, args.loss),
    #                                   (args.queue, args.queue),
    #                                   constant_bw=not args.time_variant_bw)]
    # for _, trace in enumerate(test_traces):
    #     rewards, _ = test_on_trace(trace, args.save_dir, args.seed)
    #     # print(np.mean(np.array(rewards)))


if __name__ == "__main__":
    # main()
    plot_pantheon_traces()
