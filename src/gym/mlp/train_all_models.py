import os
from multiprocessing import Process

default_hist = 10
default_arch = "32,16"
default_gamma = 0.99

all_hists = [1, 2, 3, 5, 10]
all_archs = ["", "16", "32,16", "64,32,16"]
all_gammas = [0.00, 0.50, 0.99]

def get_model_name(hist, arch, gamma, replica):
    return "model_%dhist_%sarch_%fgamma_run_%d_very_high_thpt_good_arch" % (hist, arch, gamma, replica)

def get_log_name(hist, arch, gamma, replica):
    return "%s_train_log.txt" % get_model_name(hist, arch, gamma, replica)

def get_model_cmd(hist, arch, gamma, replica):
    return "python3 stable_solve.py --history-len=%d --arch=%s --gamma=%f --model-dir=%s > %s" % (
        hist, arch, gamma, get_model_name(hist, arch, gamma, replica), get_log_name(hist, arch, gamma, replica))

cmds = []
n_replicas = 3

for i in range(1, n_replicas + 1):
    for hist in all_hists:
        cmd = get_model_cmd(hist, default_arch, default_gamma, i)
        if cmd not in cmds:
            cmds.append(cmd)

    for arch in all_archs:
        cmd = get_model_cmd(default_hist, arch, default_gamma, i)
        if cmd not in cmds:
            cmds.append(cmd)

    for gamma in all_gammas:
        cmd = get_model_cmd(default_hist, default_arch, gamma, i)
        if cmd not in cmds:
            cmds.append(cmd)

def run_training(cmd):
    os.system(cmd)

procs = []
for cmd in cmds:
    proc = Process(target=run_training, args=(cmd,))
    procs.append(proc)
    proc.start()

for proc in procs:
    proc.join()

print("All training finished!")
