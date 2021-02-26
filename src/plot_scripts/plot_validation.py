import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.figure()
for bw in [100, 1000, 1500, 2000, 3000, 5000]:
        df = pd.read_csv('../../results/rand_bw/bw_50_{}/validation_log.csv'.format(bw))
        plt.plot(df['n_calls'], df['mean_validation_reward'], 'o-', label=str(bw))
plt.legend()
#
# fig, axes = plt.subplots(2, 5, figsize=(20, 6))
#
# for i, ax in zip(range(10), axes.flatten()):
#     df = pd.read_csv('../../results/tmp/pcc_env_log_run_{}.csv'.format(i))
#     plt.plot()

# for step in [10000, 100000, 500000, 1000000, 2000000]:
plt.figure()
for seed in [42, 43, 44, 45, 46]:
    for step in [5000000]:
        df = pd.read_csv('../../results/train_{}_steps_fixed_seed_{}_fixed_send_rate/validation_log.csv'.format(step, seed))
        plt.plot(df['n_calls'], df['mean_validation_reward'], 'o-', label=str(step) + ', seed ' + str(seed))
plt.legend()

# fig, axes = plt.subplots(2, 5, figsize=(20, 6))
#
# for i, ax in zip(range(10), axes.flatten()):
#     df = pd.read_csv('../../results/tmp/pcc_env_log_run_{}.csv'.format(i))
#     plt.plot()


# plt.figure()
# avg_test_rewards = []
# for step in [10000, 100000, 500000, 1000000, 2000000]:
#     results_files = glob.glob('../../results/train_{}_steps/rl_test/*.csv'.format(step))
#     rewards = []
#     for result_file in results_files:
#         df = pd.read_csv(result_file)
#         rewards.extend(df['reward'].values)
#     avg_test_rewards.append(np.mean(np.array(rewards)))
#
#     print(results_files)
# plt.plot([10000, 100000, 500000, 1000000, 2000000], avg_test_rewards, 'o-')


plt.show()
