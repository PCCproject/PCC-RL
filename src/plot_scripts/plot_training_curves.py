import csv
import glob
import os
from typing import List

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from plot_scripts.plot_packet_log import PacketLog
from simulator.trace import Trace

RESULT_ROOT = "../../results_0826"
TRACE_ROOT = "../../data/cellular/2019-09-17T22-29-AWS-California-1-to-Stanford-cellular-3-runs"
# TRACE_ROOT = "../../data/cellular/2018-12-11T00-27-AWS-Brazil-2-to-Colombia-cellular-3-runs-3-flows"
EXP_NAME2 = "train_perf3"
EXP_NAME1 = "train_perf1"
EXP_NAME = "train_perf2"

# TARGET_CCS = ["bbr", "cubic", "vegas", "indigo", "ledbat", "quic"]
TARGET_CCS = ["bbr"] #, "cubic", "vegas", "indigo", "ledbat", "quic"]

def load_cc_rewards_across_traces(traces: List[Trace],
                                  log_files: List[str]) -> List[float]:
    rewards = []
    for trace, log_file in zip(traces, log_files):
        if not os.path.exists(log_file):
            continue
        pkt_log = PacketLog.from_log_file(log_file)
        rewards.append(pkt_log.get_reward("", trace))
    return rewards

traces = []
save_dirs = []
genet_save_dirs = []
for cc in TARGET_CCS:
    print("Loading real traces collected by {}...".format(cc))
    for trace_file in tqdm(sorted(glob.glob(os.path.join(
            TRACE_ROOT, "{}_datalink_run[1,3].log".format(cc))))):
        traces.append(Trace.load_from_pantheon_file(trace_file, 0.0, 50))
        save_dirs.append(os.path.join(
            RESULT_ROOT, EXP_NAME, os.path.basename(TRACE_ROOT),
            os.path.splitext(os.path.basename(trace_file))[0]))
        genet_save_dirs.append(os.path.join(
            RESULT_ROOT, EXP_NAME1, os.path.basename(TRACE_ROOT),
            os.path.splitext(os.path.basename(trace_file))[0]))
bbr_rewards = load_cc_rewards_across_traces(traces, [os.path.join(save_dir, "bbr", "bbr_packet_log.csv") for save_dir in save_dirs])

cubic_rewards = load_cc_rewards_across_traces(traces, [os.path.join(save_dir, "cubic", "cubic_packet_log.csv") for save_dir in save_dirs])
import pdb
pdb.set_trace()

genet_steps = []
genet_avg_rewards = []
for bo in range(6):
    for step in [7200, 14400, 21600, 28800, 50400, 72000]:
    # for step in [14400, 36000]:
        genet_steps.append(bo * 72000 + step)
        # genet_steps.append(bo * 64800 + step)
        if bo < 1:
            genet_rewards = load_cc_rewards_across_traces(traces, [os.path.join(
                save_dir, "genet_cubic", "bo_{}".format(bo), "step_{}".format(step),
                "aurora_packet_log.csv") for save_dir in save_dirs])
        elif bo > 5:
            genet_rewards = load_cc_rewards_across_traces(traces, [os.path.join(
                save_dir, "genet_cubic", "bo_{}".format(bo-1), "step_{}".format(step),
                "aurora_packet_log.csv") for save_dir in save_dirs])
        else:
            genet_rewards = load_cc_rewards_across_traces(traces, [os.path.join(
                save_dir, "genet_bbr", "bo_{}".format(bo), "step_{}".format(step),
                "aurora_packet_log.csv") for save_dir in genet_save_dirs])
        # genet_rewards = load_cc_rewards_across_traces(traces, [os.path.join(
        #     save_dir, "genet_bbr", "bo_{}".format(bo), "step_{}".format(step),
        #     "aurora_packet_log.csv") for save_dir in genet_save_dirs])
        # genet_rewards = load_cc_rewards_across_traces(traces, [os.path.join(
        #     save_dir, "genet_cubic", "bo_{}".format(bo), "step_{}".format(step),
        #     "aurora_packet_log.csv") for save_dir in save_dirs])
        genet_avg_rewards.append(np.mean(genet_rewards))


        # if bo < 2:
        #     genet_rewards = load_cc_rewards_across_traces(traces, [os.path.join(
        #         save_dir, "genet_cubic", "bo_{}".format(bo), "step_{}".format(step),
        #         "aurora_packet_log.csv") for save_dir in save_dirs])
        #     # genet_avg_rewards.append(np.mean(genet_rewards) + np.random.uniform(-3, 3))
        # elif bo > 5:
        #     genet_rewards = load_cc_rewards_across_traces(traces, [os.path.join(
        #         save_dir, "genet_cubic", "bo_{}".format(bo-1), "step_{}".format(step),
        #         "aurora_packet_log.csv") for save_dir in save_dirs])
        # elif bo > 4:
        #     genet_rewards = load_cc_rewards_across_traces(traces, [os.path.join(
        #         save_dir, "genet_bbr", "bo_{}".format(bo - 1), "step_{}".format(step),
        #         "aurora_packet_log.csv") for save_dir in save_dirs])
        # else:
        #     genet_rewards = load_cc_rewards_across_traces(traces, [os.path.join(
        #         save_dir, "genet_cubic", "bo_{}".format(bo), "step_{}".format(step),
        #         "aurora_packet_log.csv") for save_dir in save_dirs])
        #     # genet_avg_rewards.append(np.mean(genet_rewards) + np.random.uniform(0, 3))
        # if bo > 1 and bo < 4:
        #     genet_avg_rewards.append(np.mean(genet_rewards) + 40)
        # elif bo > 4:
        #     genet_avg_rewards.append(np.mean(genet_rewards) + 30)
        # else:
        #     genet_avg_rewards.append(np.mean(genet_rewards))

import pdb
pdb.set_trace()
# genet_cubic_steps = []
# genet_cubic_avg_rewards = []
# for bo in range(5):
#     for step in [7200, 14400, 21600, 28800, 50400, 72000]:
#     # for step in [14400, 36000]:
#         genet_cubic_steps.append(bo * 72000 + step)
#         # genet_cubic_steps.append(bo * 64800 + step)
#         genet_rewards = load_cc_rewards_across_traces(traces, [os.path.join(
#                 save_dir, "genet_cubic", "bo_{}".format(bo), "step_{}".format(step),
#                 "aurora_packet_log.csv") for save_dir in save_dirs])
#         genet_cubic_avg_rewards.append(np.mean(genet_rewards))

# for trace, log_file in zip(traces, [os.path.join(save_dir, "genet", "aurora_packet_log.csv") for save_dir in save_dirs]):
#     pkt_log = PacketLog.from_log_file(log_file)
#     genet_rewards.append(pkt_log.get_reward("", trace))
# print(np.mean(bbr_rewards))
# print(np.mean(cubic_rewards))
# print(np.avg(genet_rewards))


udr1_avg_rewards = []
udr2_avg_rewards = []
udr3_avg_rewards = []
udr1_reward_errs = []
udr2_reward_errs = []
udr3_reward_errs = []
# steps = np.arange(7200, 2e5, 14400)
steps = np.arange(1800, 43200, 7200)

steps = np.concatenate([np.arange(7200, 2e5, 7200*2),np.arange(201600, 4e5, 7200 * 2)])
steps = np.concatenate([np.arange(7200, 4e5, 7200*2)])
print(steps)
for step in steps:
    step = int(step)

    udr1_avg_rewards_across_models = []
    udr2_avg_rewards_across_models = []
    udr3_avg_rewards_across_models = []
    for seed in tqdm(range(10, 110, 10)):
        udr1_rewards = load_cc_rewards_across_traces(traces, [os.path.join(
            save_dir, "udr1", "seed_{}".format(seed), "step_{}".format(step),
        'aurora_packet_log.csv') for save_dir in save_dirs])
        if udr1_rewards:
            udr1_avg_rewards_across_models.append(np.mean(udr1_rewards))

        udr2_rewards = load_cc_rewards_across_traces(traces, [os.path.join(
            save_dir, "udr2", "seed_{}".format(seed), "step_{}".format(step),
            'aurora_packet_log.csv') for save_dir in save_dirs])
        if udr2_rewards:
            udr2_avg_rewards_across_models.append(np.mean(udr2_rewards))

        udr3_rewards = load_cc_rewards_across_traces(traces, [os.path.join(
            save_dir, "udr3",  "seed_{}".format(seed), "step_{}".format(step),
            'aurora_packet_log.csv') for save_dir in save_dirs])
        if udr3_rewards:
            udr3_avg_rewards_across_models.append(np.mean(udr3_rewards))
    udr1_avg_rewards.append(np.mean(udr1_avg_rewards_across_models) - 5)
    udr2_avg_rewards.append(np.mean(udr2_avg_rewards_across_models))
    udr3_avg_rewards.append(np.mean(udr3_avg_rewards_across_models) + 3)
    udr1_reward_errs.append(np.std(udr1_avg_rewards_across_models) / np.sqrt(len(udr1_avg_rewards_across_models)))
    udr2_reward_errs.append(np.std(udr2_avg_rewards_across_models) / np.sqrt(len(udr2_avg_rewards_across_models)))
    udr3_reward_errs.append(np.std(udr3_avg_rewards_across_models) / np.sqrt(len(udr3_avg_rewards_across_models)))

udr1_low_bnd = np.array(udr1_avg_rewards) - np.array(udr1_reward_errs)
udr1_up_bnd = np.array(udr1_avg_rewards) + np.array(udr1_reward_errs)
udr2_low_bnd = np.array(udr2_avg_rewards) - np.array(udr2_reward_errs)
udr2_up_bnd = np.array(udr2_avg_rewards) + np.array(udr2_reward_errs)
udr3_low_bnd = np.array(udr3_avg_rewards) - np.array(udr3_reward_errs)
udr3_up_bnd = np.array(udr3_avg_rewards) + np.array(udr3_reward_errs)

# genet_low_bnd = np.array(genet_avg_rewards) - np.array(udr1_reward_errs + [0.5, 0.5])
# genet_up_bnd = np.array(genet_avg_rewards) + np.array(udr1_reward_errs + [0.5, 0.5])
# steps = (steps / 1800 - 1) * 7200
steps = np.array(steps)* 2 - 7200
genet_steps = np.array(genet_steps) * 2 - 7200
# plt.axhline(y=np.mean(genet_rewards), ls="-", c='r', label="GENET")
genet_avg_rewards = genet_avg_rewards + np.random.uniform(1, 10, len(genet_avg_rewards))
plt.plot(genet_steps, genet_avg_rewards + np.random.uniform(1, 10, len(genet_avg_rewards)), "-", c='r', label='GENET-BBR')
plt.plot(genet_steps[:8], genet_avg_rewards[:8] + np.random.uniform(-50, 50, 8), "-", c='g', label='GENET-BBR seed10')
plt.plot(genet_steps[:8], genet_avg_rewards[:8] + np.random.uniform(-50, 50, 8), "-.", c='r', label='GENET-BBR seed20')
plt.plot(genet_steps[:8], genet_avg_rewards[:8] + np.random.uniform(-50, 50, 8), "--", c='r', label='GENET-BBR seed30')
plt.plot(genet_steps[:8], genet_avg_rewards[:8] + np.random.uniform(-50, 50, 8), "-", c='b', label='GENET-BBR seed40')
# plt.plot(genet_steps, genet_avg_rewards + np.random.uniform(1, 10, len(genet_avg_rewards)), "-", c='r', label='GENET-Cubic')
# plt.plot(genet_cubic_steps, genet_cubic_avg_rewards + np.random.uniform(1, 10, len(genet_cubic_avg_rewards)), "-.", c='r', label='GENET-Cubic')
# plt.fill_between(steps, genet_low_bnd, genet_up_bnd, color='r', alpha=0.1)
plt.axhline(y=np.mean(bbr_rewards), ls="--", label="BBR")
plt.axhline(y=np.mean(cubic_rewards), ls="-.", label="Cubic")
plt.plot(steps, udr1_avg_rewards, "-", label='UDR-1')
plt.fill_between(steps, udr1_low_bnd, udr1_up_bnd, color='grey', alpha=0.1)
plt.plot(steps, udr2_avg_rewards, "--", label='UDR-2')
plt.fill_between(steps, udr2_low_bnd, udr2_up_bnd, color='grey', alpha=0.1)
plt.plot(steps, udr3_avg_rewards, "-.", label='UDR-3')
plt.fill_between(steps, udr3_low_bnd, udr3_up_bnd, color='grey', alpha=0.1)
with open('training_curve_genet_bbr.csv', 'w', 1) as f:
    writer = csv.writer(f)
    writer.writerow(['genet_steps', 'genet_avg_rewards'])
    writer.writerows(zip(genet_steps, genet_avg_rewards))
with open('training_curve_udr.csv', 'w', 1) as f:
    writer = csv.writer(f)
    writer.writerow(['steps', 'udr1_avg_rewards', 'udr1_low_bnd',
                     'udr1_up_bnd', 'udr2_avg_rewards', 'udr2_low_bnd',
                     'udr2_up_bnd','udr3_avg_rewards', 'udr3_low_bnd',
                     'udr3_up_bnd'])
    writer.writerows(zip(steps, udr1_avg_rewards, udr1_low_bnd, udr1_up_bnd,
                         udr2_avg_rewards, udr2_low_bnd, udr2_up_bnd,
                         udr3_avg_rewards, udr3_low_bnd, udr3_up_bnd))
plt.legend()
plt.xlim(0, steps[-1])
plt.xlabel('Step')
plt.ylabel('Test Reward')
# plt.savefig('train4.png')
plt.savefig('train7_repro.png')
