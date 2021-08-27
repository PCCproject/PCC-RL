import glob
import os

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from plot_scripts.plot_packet_log import PacketLog
from simulator.trace import Trace

RESULT_ROOT = "../../results"
TRACE_ROOT = "../../data/2019-09-17T22-29-AWS-California-1-to-Stanford-cellular-3-runs"
EXP_NAME = "train_perf"

TARGET_CCS = ["bbr", "cubic", "vegas", "indigo", "ledbat", "quic"]

traces = []
save_dirs = []
bbr_rewards = []
cubic_rewards = []
genet_rewards = []
for cc in TARGET_CCS:
    print("Loading real traces collected by {}...".format(cc))
    for trace_file in tqdm(sorted(glob.glob(os.path.join(
            TRACE_ROOT, "{}_datalink_run[1-3].log".format(cc))))):
        traces.append(Trace.load_from_pantheon_file(trace_file, 0.0, 50))
        save_dirs.append(os.path.join(
            RESULT_ROOT, EXP_NAME, os.path.basename(TRACE_ROOT),
            os.path.splitext(os.path.basename(trace_file))[0]))

for trace, log_file in zip(traces, [os.path.join(save_dir, "bbr", "bbr_packet_log.csv") for save_dir in save_dirs]):
    pkt_log = PacketLog.from_log_file(log_file)
    bbr_rewards.append(pkt_log.get_reward("", trace))

for trace, log_file in zip(traces, [os.path.join(save_dir, "cubic", "cubic_packet_log.csv") for save_dir in save_dirs]):
    pkt_log = PacketLog.from_log_file(log_file)
    cubic_rewards.append(pkt_log.get_reward("", trace))

for trace, log_file in zip(traces, [os.path.join(save_dir, "genet", "aurora_packet_log.csv") for save_dir in save_dirs]):
    pkt_log = PacketLog.from_log_file(log_file)
    genet_rewards.append(pkt_log.get_reward("", trace))
print(np.mean(bbr_rewards))
print(np.mean(cubic_rewards))
print(np.mean(genet_rewards))

udr1_mean_rewards = []
udr2_mean_rewards = []
udr3_mean_rewards = []
udr1_reward_errs = []
udr2_reward_errs = []
udr3_reward_errs = []
# steps = np.arange(7200, 2e5, 14400)
steps = np.arange(1800, 43200, 7200)
for step in steps:
    step = int(step)
    udr1_rewards = []
    udr2_rewards = []
    udr3_rewards = []
    for trace, log_file in zip(traces, [os.path.join(
        save_dir, "udr1", "step_{}".format(step), 'aurora_packet_log.csv') for save_dir in save_dirs]):
        pkt_log = PacketLog.from_log_file(log_file)
        udr1_rewards.append(pkt_log.get_reward("", trace))
    for trace, log_file in zip(traces, [os.path.join(
        save_dir, "udr2", "step_{}".format(step), 'aurora_packet_log.csv') for save_dir in save_dirs]):
        pkt_log = PacketLog.from_log_file(log_file)
        udr2_rewards.append(pkt_log.get_reward("", trace))
    for trace, log_file in zip(traces, [os.path.join(
        save_dir, "udr3", "step_{}".format(step), 'aurora_packet_log.csv') for save_dir in save_dirs]):
        pkt_log = PacketLog.from_log_file(log_file)
        udr3_rewards.append(pkt_log.get_reward("", trace))
    udr1_mean_rewards.append(np.mean(udr1_rewards))
    udr2_mean_rewards.append(np.mean(udr2_rewards))
    udr3_mean_rewards.append(np.mean(udr3_rewards))
    udr1_reward_errs.append(np.std(udr1_rewards) / np.sqrt(len(udr1_rewards)))
    udr2_reward_errs.append(np.std(udr2_rewards) / np.sqrt(len(udr2_rewards)))
    udr3_reward_errs.append(np.std(udr3_rewards) / np.sqrt(len(udr3_rewards)))

udr1_low_bnd = np.array(udr1_mean_rewards) - np.array(udr1_reward_errs)
udr1_up_bnd = np.array(udr1_mean_rewards) + np.array(udr1_reward_errs)
udr2_low_bnd = np.array(udr2_mean_rewards) - np.array(udr2_reward_errs)
udr2_up_bnd = np.array(udr2_mean_rewards) + np.array(udr2_reward_errs)
udr3_low_bnd = np.array(udr3_mean_rewards) - np.array(udr3_reward_errs)
udr3_up_bnd = np.array(udr3_mean_rewards) + np.array(udr3_reward_errs)

steps = (steps / 1800 - 1) * 7200
plt.axhline(y=np.mean(genet_rewards), ls="-", c='r', label="GENET")
plt.axhline(y=np.mean(bbr_rewards), ls="--", label="BBR")
plt.axhline(y=np.mean(cubic_rewards), ls="-.", label="Cubic")
plt.plot(steps, udr1_mean_rewards, "-", label='UDR-1')
plt.fill_between(steps, udr1_low_bnd, udr1_up_bnd, color='grey', alpha=0.1)
plt.plot(steps, udr2_mean_rewards, "--", label='UDR-2')
plt.fill_between(steps, udr2_low_bnd, udr2_up_bnd, color='grey', alpha=0.1)
plt.plot(steps, udr3_mean_rewards, "-.", label='UDR-3')
plt.fill_between(steps, udr3_low_bnd, udr3_up_bnd, color='grey', alpha=0.1)
plt.legend()
# plt.xlim(0, 0.2 * 1e6)
plt.xlabel('Step')
plt.ylabel('Test Reward')
plt.savefig('train.png')
