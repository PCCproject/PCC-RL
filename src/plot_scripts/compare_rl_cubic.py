import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from utils import read_json_file, write_json_file
ROOT = '../../results'
exp_name = 'rand_bw'
# labels = ['bw_1_100', 'bw_1_500', 'bw_1_1000', 'bw_1_2000', 'bw_1_5000']
labels = ['bw_50_100', 'bw_50_500', 'bw_50_1000', 'bw_50_1500', 'bw_50_2000', 'bw_50_3000'] #, 'bw_1_5000']
# num_traces = [3, 5, 7, 8, 9, 10, 12]
num_traces = [10, 10, 10, 10, 10, 10] #, 10]
mean_rl_rewards = []
mean_cubic_rewards = []
rl_errs = []
cubic_errs = []
bw_list = [50, 80, 100, 300, 500, 800, 1000, 1500, 2000, 3000]#, 4000]#, 5000]
for range_name, num_trace in zip(labels, num_traces):
    rl_rewards = []
    cubic_rewards = []
    rl_normalized_rewards = []
    cubic_normalized_rewards = []
    for i in range(len(bw_list)): #num_trace):
        # rl_log = read_json_file('../../results/test_rl_in_dist/rl_test_log{}.json'.format(i))
        # rl_rewards.extend([event['Reward'] for event in rl_log["Events"]])
        # cubic_log = read_json_file('../../results/test_cubic_in_dist/cubic_test_log{}.json'.format(i))
        # cubic_rewards.extend([event['Reward'] for event in cubic_log["Events"]])

        rl_log = pd.read_csv(os.path.join(ROOT, exp_name, range_name, 'rl_test', 'rl_test_log{}.csv'.format(i)))
        print(rl_log)
        rl_rewards.extend(rl_log['reward'].values)
        cubic_log = pd.read_csv(os.path.join(ROOT, exp_name, labels[0], 'cubic_test', 'cubic_test_log{}.csv'.format(i)))
        cubic_rewards.extend(cubic_log['reward'].values)
        rl_normalized_rewards.extend((rl_log['throughput'] / bw_list[i] * 10 - 1000 * rl_log['latency'] / 0.05 * 1000 - rl_log['loss'] * 2000).values)
        cubic_normalized_rewards.extend((cubic_log['throughput'] / bw_list[i] * 10 - 1000 * cubic_log['latency'] / 0.05 * 1000 - cubic_log['loss'] * 2000).values)
    # mean_rl_rewards.append(np.mean(np.array(rl_rewards)))
    # mean_cubic_rewards.append(np.mean(np.array(cubic_rewards)))
    mean_rl_rewards.append(np.mean(np.array(rl_normalized_rewards)))
    mean_cubic_rewards.append(np.mean(np.array(cubic_normalized_rewards)))
    # rl_errs.append(np.std(np.array(rl_rewards))/np.sqrt(len(rl_rewards)))
    # cubic_errs.append(np.std(np.array(cubic_rewards))/np.sqrt(len(cubic_rewards)))
    rl_errs.append(np.std(np.array(rl_normalized_rewards))/np.sqrt(len(rl_rewards)))
    cubic_errs.append(np.std(np.array(cubic_normalized_rewards))/np.sqrt(len(cubic_rewards)))

# plt.show()
# import sys
# sys.exit()


plt.figure()
ax = plt.gca()
ax.errorbar(np.arange(len(labels)), np.array(mean_rl_rewards), yerr=rl_errs, fmt='-o', label='Aurora')
ax.errorbar(np.arange(len(labels)), np.array(mean_cubic_rewards), yerr=cubic_errs, fmt='-o', label='Cubic')
# ax.errorbar(np.arange(len(labels)), np.array(rl_normalized_rewards),  fmt='-o', label='Aurora')
# ax.errorbar(np.arange(len(labels)), np.array(cubic_normalized_rewards),  fmt='-o', label='Cubic')
plt.xticks(np.arange(len(labels)), labels, rotation=15)
# plt.ylabel('Rewards')
plt.ylabel('Normalized Rewards')
plt.legend()
# plt.show()
# plt.title('Aurora vs. Cubic on envs share the same dist with training envs')
# plt.title("In-distribution")


# rl_rewards = []
# cubic_rewards = []
# mean_rl_rewards = []
# mean_cubic_rewards = []
# for i in range(6):
#     # rl_log = read_json_file('../../results/test_rl_out_dist/rl_test_log{}.json'.format(i))
#     # rl_rewards.extend([event['Reward'] for event in rl_log["Events"]])
#     # cubic_log = read_json_file('../../results/test_cubic_out_dist/cubic_test_log{}.json'.format(i))
#     # cubic_rewards.extend([event['Reward'] for event in cubic_log["Events"]])
#     rl_log = pd.read_csv('../../results/test_rl_out_dist/rl_test_log{}.csv'.format(i), header=None)
#     rl_rewards.extend(rl_log[6].values)
#     mean_rl_rewards.append(rl_log[6].mean())
#     # cubic_log = pd.read_csv('../../results/test_cubic_out_dist/cubic_test_log{}.csv'.format(i), header=None)
#     cubic_log = pd.read_csv('./tmp/cubic_test_log{}.csv'.format(i), header=None)
#     cubic_rewards.extend(cubic_log[6].values)
#     mean_cubic_rewards.append(cubic_log[6].mean())
#     if i == 1:
#         plt.figure()
#         plt.plot(cubic_log[0], cubic_log[1])
#         plt.figure()
#         plt.plot(cubic_log[0], cubic_log[5])
#         plt.ylabel('throughout')
#         plt.show()
# # print(len(cubic_rewards))
# # print(len(rl_rewards))
# # plt.figure()
# # plt.bar([0, 1], [np.mean(rl_rewards), np.mean(cubic_rewards)])
# # plt.xticks([0, 1], ['rl', 'cubic'])
# # plt.ylabel('Rewards')
# # plt.title("Out-of-distribution")
# #
# # plt.figure()
# # plt.plot(np.arange(len(mean_rl_rewards)), mean_rl_rewards, 'o-', label='rl')
# # plt.plot(np.arange(len(mean_cubic_rewards)), mean_cubic_rewards, 'o-', label='cubic')
# # plt.legend()
# # plt.ylabel('rewards')
# # plt.xlabel('config id')
ROOT = '../../results'
exp_name = 'rand_queue'
# labels = ['bw_1_100', 'bw_1_500', 'bw_1_1000', 'bw_1_2000', 'bw_1_5000']
labels = ['queue_0_0','queue_0_1', 'queue_0_2', 'queue_0_3', 'queue_0_4', 'queue_0_5', 'queue_0_6', 'queue_0_7', 'queue_0_8'] #, 'bw_1_5000']
updated_labels = ["queue_2_"+ str(int(np.exp(idx)+1)) for idx, label in enumerate(labels)] #, 'bw_1_5000']
num_traces = [1, 5, 7, 8, 9, 10, 12]
mean_rl_rewards = []
mean_cubic_rewards = []
rl_errs = []
cubic_errs = []

queue_list = [0, 1, 2, 3, 4, 5,  6, 7, 8]
num_traces = [len(queue_list)] * 9
plt.figure()
for range_name, num_trace in zip(labels, num_traces):
    rl_rewards = []
    cubic_rewards = []
    # num_trace = 9
    for i in range(num_trace):
        # rl_log = read_json_file('../../results/test_rl_in_dist/rl_test_log{}.json'.format(i))
        # rl_rewards.extend([event['Reward'] for event in rl_log["Events"]])
        # cubic_log = read_json_file('../../results/test_cubic_in_dist/cubic_test_log{}.json'.format(i))
        # cubic_rewards.extend([event['Reward'] for event in cubic_log["Events"]])

        rl_log = pd.read_csv(os.path.join(ROOT, exp_name, range_name, 'rl_test', 'rl_test_log{}.csv'.format(i)))
        rl_rewards.extend(rl_log['reward'].values)
        cubic_log = pd.read_csv(os.path.join(ROOT, exp_name, labels[0], 'cubic_test', 'cubic_test_log{}.csv'.format(i)))
        cubic_rewards.extend(cubic_log['reward'].values)
    mean_rl_rewards.append(np.mean(np.array(rl_rewards)))
    mean_cubic_rewards.append(np.mean(np.array(cubic_rewards)))
    rl_errs.append(np.std(np.array(rl_rewards))/np.sqrt(len(rl_rewards)))
    cubic_errs.append(np.std(np.array(cubic_rewards))/np.sqrt(len(cubic_rewards)))

ax = plt.gca()
print(len(labels), len(mean_rl_rewards))
print(len(labels), len(mean_cubic_rewards))
ax.errorbar(np.arange(len(labels)), np.array(mean_rl_rewards), yerr=rl_errs, fmt='-o', label='Aurora')
ax.errorbar(np.arange(len(labels)), np.array(mean_cubic_rewards), yerr=cubic_errs, fmt='-o', label='Cubic')
plt.xticks(np.arange(len(labels)), updated_labels, rotation=15)
plt.ylabel('Rewards')
plt.legend()

# plt.figure()
# ax = plt.gca()
# ax.errorbar(np.arange(len(labels)), np.array(mean_rl_rewards), yerr=rl_errs, fmt='-o', label='Aurora')
# ax.errorbar(np.arange(len(labels)), np.array(mean_cubic_rewards), yerr=cubic_errs, fmt='-o', label='Cubic')
# plt.xticks(np.arange(len(labels)), labels, rotation=15)
# plt.ylabel('Rewards')
# plt.legend()
# plt.title('Aurora vs. Cubic on envs share the same dist with training envs')

plt.show()
