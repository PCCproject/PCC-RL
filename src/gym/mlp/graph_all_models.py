import os
import numpy as np
from scipy.signal import savgol_filter

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

default_hist = 10
default_arch = "32,16"
default_gamma = 0.99

all_hists = [1, 2, 3, 5, 10]
all_archs = ["", "16", "32,16", "64,32,16"]
all_gammas = [0.00, 0.50, 0.99]
#all_gammas = []

smoothness = 51
n_replicas = 3

def get_model_name(hist, arch, gamma, replica):
    return "model_%dhist_%sarch_%fgamma_run_%d_very_high_thpt_good_arch" % (hist, arch, gamma, replica)

def get_log_name(hist, arch, gamma, replica):
    return "%s_train_log.txt" % get_model_name(hist, arch, gamma, replica)

def get_model_cmd(hist, arch, gamma, replica):
    return "python3 stable_solve.py --history-len=%d --arch=%s --gamma=%f --model-dir=%s > %s" % (
        hist, arch, gamma, get_model_name(hist, arch, gamma, replica), get_log_name(hist, arch, gamma, replica))

def get_reward_lines(all_lines):
    reward_lines = []
    for line in all_lines:
        if "Reward" in line and "Ewma Reward" in line:
            reward_lines.append(line)
    return reward_lines

def get_line_reward(line):
    return float(line.split(",")[0].split(" ")[-1])

def get_model_rewards(hist, arch, gamma, replica):
    all_lines = None
    with open(get_log_name(hist, arch, gamma, replica)) as f:
        all_lines = f.readlines()
    reward_lines = get_reward_lines(all_lines)
    rewards = [get_line_reward(line) for line in reward_lines]
    mean_rew = np.mean(rewards)
    if (len(rewards) > 500):
        mean_rew = np.mean(rewards[-500:])
    print("%s reward %f" % (get_model_name(hist, arch, gamma, replica), mean_rew))
    return rewards

def trim_to_equal_length(list_of_lists):
    final_len = min([len(sublist) for sublist in list_of_lists])
    return [sublist[:final_len] for sublist in list_of_lists]

def average_sublists(list_of_lists):
    arr = np.array(trim_to_equal_length(list_of_lists))
    return np.mean(arr, axis=0)

def get_model_avg_rewards(hist, arch, gamma):
    global n_replicas
    all_rews = [get_model_rewards(hist, arch, gamma, replica) for replica in range(1, n_replicas + 1)]
    print("Mean rew for %s: %f (%d samples done)" % (get_model_name(hist, arch, gamma, 0), np.mean(average_sublists(all_rews)[-500:]), len(average_sublists(all_rews))))
    return average_sublists(all_rews)


for hist in all_hists:
    rews = get_model_avg_rewards(hist, default_arch, default_gamma)
    plt.plot(range(0, len(rews)), savgol_filter(rews, smoothness, 0), label="History %d" % hist)

plt.legend()
plt.title("Effects of History Length")
plt.xlabel("Training Episode")
plt.ylabel("Training Reward")
plt.savefig("history_rewards.png")
plt.close()

for arch in all_archs:
    rews = get_model_avg_rewards(default_hist, arch, default_gamma)
    plt.plot(range(0, len(rews)), savgol_filter(rews, smoothness, 0), label="Arch %s" % arch)

plt.legend()
plt.title("Effects of Model Architecture")
plt.xlabel("Training Episode")
plt.ylabel("Training Reward")
plt.savefig("arch_rewards.png")
plt.close()

for gamma in all_gammas:
    rews = get_model_avg_rewards(default_hist, default_arch, gamma)
    plt.plot(range(0, len(rews)), savgol_filter(rews, smoothness, 0), label="Gamma %f" % gamma)

plt.legend()
plt.title("Effects of Discount Factor")
plt.xlabel("Training Episode")
plt.ylabel("Training Reward")
plt.savefig("gamma_rewards.png")
plt.close()
