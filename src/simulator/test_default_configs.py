import os
import csv
import pandas as pd
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common.utils import set_seed, read_json_file
from simulator.aurora import Aurora
from simulator.evaluate_cubic import test_on_trace as test_cubic_on_trace
from simulator.trace import generate_trace
from plot_scripts.plot_packet_log import PacketLog

plt.style.use('seaborn-deep')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams["figure.figsize"] = (10,6)

# cmd = "python evaluate_cubic.py --bandwidth {} --delay {} --queue {} " \
#     "--loss {} --duration {} --save-dir {}".format(
#         bandwidth, delay, int(queue), loss, 10, "tmp")
# print(cmd)
# subprocess.check_output(cmd, shell=True).strip()
UDR_BIG_MODEL_PATH = "/tank/zxxia/PCC-RL/results_0503/udr_7_dims/udr_large/seed_50/model_step_396000.ckpt"
UDR_MID_MODEL_PATH = "/tank/zxxia/PCC-RL/results_0503/udr_7_dims/udr_mid/seed_40/model_step_288000.ckpt"
UDR_SMALL_MODEL_PATH = "/tank/zxxia/PCC-RL/results_0503/udr_7_dims/udr_small/seed_50/model_step_1800000.ckpt"
# MODEL_PATH = "/tank/zxxia/PCC-RL/results_0503/bo_new/seed_10/bo_0/model_step_36000.ckpt"
# MODEL_PATH = "/tank/zxxia/PCC-RL/results_0503/bo_delay/seed_10/bo_0/model_step_28800.ckpt"
MODEL_PATH = "/tank/zxxia/PCC-RL/results_0503/bo_bandwidth/seed_10/bo_10/model_step_36000.ckpt"
# MODEL_PATH = "/tank/zxxia/PCC-RL/results_0503/bo_delay_new/seed_10/bo_10/model_step_36000.ckpt"
# MODEL_PATH = "/tank/zxxia/PCC-RL/results_0503/bo_delay_new_no_delay_noise/seed_10/bo_4/model_step_36000.ckpt"
# MODEL_PATH = "/tank/zxxia/PCC-RL/results_0503/bo_delay_new_no_delay_noise_fix_seed/seed_50/bo_1/model_step_21600.ckpt"
# MODEL_PATH = "/tank/zxxia/PCC-RL/results_0503/bo_delay_new_no_delay_noise_fix_seed_fix_reward/seed_50/bo_0/model_step_36000.ckpt"
MODEL_PATH = "/tank/zxxia/PCC-RL/results_0503/bo_delay_new_no_delay_noise_fix_seed_fix_reward_10s_new/seed_50/bo_2/model_step_14400.ckpt"
# MODEL_PATH = "/tank/zxxia/PCC-RL/results_0503/bo_delay_new_no_delay_noise_fix_seed_fix_reward_10s_new/seed_50/bo_5/model_step_14400.ckpt"
# MODEL_PATH = "/tank/zxxia/PCC-RL/results_0503/bo_T_s/seed_20/bo_0/model_step_50400.ckpt"
SAVE_DIR = '../../results_0503'


# metric = 'bandwidth'
metric = 'delay'
# metric = 'T_s'

set_seed(20)

DEFAULT_CONFIGS = []
for _ in range(10):
    DEFAULT_CONFIGS.append(
        (round(10 ** np.random.uniform(np.log10(1), np.log10(6), 1).item(), 2),
         round(np.random.uniform(5, 200, 1).item(), 2),
         round(np.random.uniform(0, 0.0, 1).item(), 2),
         int(10 ** np.random.uniform(np.log10(5), np.log10(30), 1).item()),
         round(np.random.randint(0, 6, 1).item(), 2),
         # round(np.random.uniform(0, 6, 1).item(), 2),
         round(np.random.uniform(0, 0, 1).item(), 2)))
# config = read_json_file("/tank/zxxia/PCC-RL/results_0503/bo_new/seed_10/bo_0.json")
# config = read_json_file("/tank/zxxia/PCC-RL/results_0503/bo_delay/seed_10/bo_0.json")
# default_configs = []
# for cf in config:
#     # default_configs.append((cf['delay'][0], cf['loss'][0], cf['queue'][0],
#     #                         cf['T_s'][0], cf['delay_noise'][0]))
#     default_configs.append((cf['bandwidth'][0], cf['loss'][0], cf['queue'][0],
#                             cf['T_s'][0], cf['delay_noise'][0]))



aurora_udr_big = Aurora(seed=20, log_dir="tmp", timesteps_per_actorbatch=10,
                        pretrained_model_path=UDR_BIG_MODEL_PATH, delta_scale=1)
aurora_udr_mid = Aurora(seed=20, log_dir="tmp", timesteps_per_actorbatch=10,
                        pretrained_model_path=UDR_MID_MODEL_PATH, delta_scale=1)
aurora_udr_small = Aurora(seed=20, log_dir="tmp", timesteps_per_actorbatch=10,
                        pretrained_model_path=UDR_SMALL_MODEL_PATH, delta_scale=1)

aurora_bo = Aurora(seed=20, log_dir="tmp", timesteps_per_actorbatch=10,
                   pretrained_model_path=MODEL_PATH, delta_scale=1)

# avg_bo_rewards = []
# avg_bo_rewards_errs = []
# avg_rewards = []
# avg_rewards_errs = []
# avg_cubic_rewards2plot = []
# avg_cubic_rewards2plot_errs = []
#
# # for config_id, (delay, loss, queue, T_s, delay_noise) in enumerate(default_configs):
# # vals = [0, 1, 2, 3, 4, 5, 6]
# vals = [5, 50, 100, 150, 200]
# all_udr_big_rewards = [[] for _ in vals]
# all_bo_rewards = [[] for _ in vals]
# all_cubic_rewards = [[] for _ in vals]


vals2test = {
    "bandwidth": [0, 1, 2, 3, 4, 5, 6],
    "delay": [5, 50, 100, 150, 200], #[25, 75, 125, 175], #
    "loss": [0, 0.01, 0.02, 0.03, 0.04, 0.05],
    "queue": [2, 10, 50, 100, 150, 200],
    "T_s": [0, 1, 2, 3, 4, 5, 6],
    "delay_noise": [0, 20, 40, 60, 80, 100],
}


for val in vals2test[metric]:
    for config_id, (bandwidth, delay, loss, queue, T_s, delay_noise) in enumerate(
            DEFAULT_CONFIGS):
        if config_id != 3 and config_id != 4 and config_id != 8:
            continue
        if metric == 'bandwidth':
            bandwidth = val
        elif metric == 'delay':
            delay = val
        elif metric == 'loss':
            loss = val
        elif metric == 'queue':
            queue = val
        elif metric == 'T_s':
            T_s = val
        elif metric == 'delay_noise':
            delay_noise = val
        else:
            raise RuntimeError

        for i in range(10):
            trace = generate_trace(duration_range=(10, 10),
                                   bandwidth_range=(1, 1+bandwidth),
                                   delay_range=(delay, delay),
                                   loss_rate_range=(loss, loss),
                                   queue_size_range=(queue, queue),
                                   T_s_range=(T_s, T_s),
                                   delay_noise_range=(
                                       delay_noise, delay_noise),
                                   constant_bw=False)

            os.makedirs(os.path.join(SAVE_DIR, 'rand_{}'.format(metric), str(val),
                                     'config_{}'.format(config_id),
                                     'trace_{}'.format(i), 'cubic'), exist_ok=True)
            os.makedirs(os.path.join(SAVE_DIR, 'rand_{}'.format(metric), str(val),
                                     'config_{}'.format(config_id),
                                     'trace_{}'.format(i), 'udr_big'), exist_ok=True)
            os.makedirs(os.path.join(SAVE_DIR, 'rand_{}'.format(metric), str(val),
                                     'config_{}'.format(config_id),
                                     'trace_{}'.format(i), 'udr_mid'), exist_ok=True)
            os.makedirs(os.path.join(SAVE_DIR, 'rand_{}'.format(metric), str(val),
                                     'config_{}'.format(config_id),
                                     'trace_{}'.format(i), 'udr_small'), exist_ok=True)
            os.makedirs(os.path.join(SAVE_DIR, 'rand_{}'.format(metric), str(val),
                                     'config_{}'.format(config_id),
                                     'trace_{}'.format(i), 'bo'), exist_ok=True)
            # cubic_rewards, cubic_pkt_log = test_cubic_on_trace(
            #     trace,
            #     os.path.join(SAVE_DIR, 'rand_{}'.format(metric), str(val),
            #                  'config_{}'.format(config_id),
            #                  'trace_{}'.format(i), 'cubic'), 20)
            #
            # ts_list, reward_list, loss_list, tput_list, delay_list, send_rate_list, \
            #     action_list, obs_list, mi_list, pkt_log = aurora_udr_big.test(
            #         trace, os.path.join(SAVE_DIR, 'rand_{}'.format(metric), str(val),
            #                             'config_{}'.format(config_id),
            #                             'trace_{}'.format(i), 'udr_big'))
            #
            # ts_list, reward_list, loss_list, tput_list, delay_list, send_rate_list, \
            #     action_list, obs_list, mi_list, pkt_log = aurora_udr_mid.test(
            #         trace, os.path.join(SAVE_DIR, 'rand_{}'.format(metric), str(val),
            #                             'config_{}'.format(config_id),
            #                             'trace_{}'.format(i), 'udr_mid'))
            # ts_list, reward_list, loss_list, tput_list, delay_list, send_rate_list, \
            #     action_list, obs_list, mi_list, pkt_log = aurora_udr_small.test(
            #         trace, os.path.join(SAVE_DIR, 'rand_{}'.format(metric), str(val),
            #                             'config_{}'.format(config_id),
            #                             'trace_{}'.format(i), 'udr_small'))
            # ts_list, reward_list, loss_list, tput_list, delay_list, send_rate_list, \
            #     action_list, obs_list, mi_list, pkt_log = aurora_bo.test(
            #         trace, os.path.join(SAVE_DIR, 'rand_{}'.format(metric), str(val),
            #                             'config_{}'.format(config_id),
            #                             'trace_{}'.format(i), 'bo'))
            # trace.dump(os.path.join(SAVE_DIR, 'rand_{}'.format(metric),
            #                         str(val), 'config_{}'.format(config_id),
            #                         'trace_{}'.format(i), 'trace.json'))

            print(metric, val, config_id, i)
avg_bo_rewards = []
avg_bo_rewards_errs = []

avg_udr_big_rewards = []
avg_udr_big_rewards_errs = []

avg_udr_mid_rewards = []
avg_udr_mid_rewards_errs = []

avg_udr_small_rewards = []
avg_udr_small_rewards_errs = []

avg_cubic_rewards = []
avg_cubic_rewards_errs = []

for config_id, (bandwidth, delay, loss, queue, T_s, delay_noise) in enumerate(
        DEFAULT_CONFIGS):
    # bo_rewards = []
    # bo_rewards_errs = []
    if config_id != 3 and config_id != 4 and config_id != 8:
        continue
    config_bo_avg_rewards = []
    config_bo_avg_rewards_errs = []
    config_udr_big_avg_rewards = []
    config_udr_big_avg_rewards_errs = []
    config_udr_mid_avg_rewards = []
    config_udr_mid_avg_rewards_errs = []
    config_udr_small_avg_rewards = []
    config_udr_small_avg_rewards_errs = []
    config_cubic_avg_rewards = []
    config_cubic_avg_rewards_errs = []
    for val in vals2test[metric]:
        if metric == 'bandwidth':
            bandwidth = val
        elif metric == 'delay':
            delay = val
        elif metric == 'loss':
            loss = val
        elif metric == 'queue':
            queue = val
        elif metric == 'T_s':
            T_s = val
        elif metric == 'delay_noise':
            delay_noise = val
        else:
            raise RuntimeError
        trace_udr_big_rewards = []
        trace_udr_mid_rewards = []
        trace_udr_small_rewards = []
        trace_bo_rewards = []
        # trace_udr_big_rewards_errs = []
        trace_cubic_rewards = []
        # trace_cubic_rewards_errs = []

        for i in range(10):
            cubic_sim_log_file = os.path.join(
                SAVE_DIR, 'rand_{}'.format(metric), str(val),
                'config_{}'.format(config_id), 'trace_{}'.format(i), 'cubic',
                'cubic_simulation_log.csv')
            cubic_pkt_log_file = os.path.join(
                SAVE_DIR, 'rand_{}'.format(metric), str(val),
                'config_{}'.format(config_id), 'trace_{}'.format(i), 'cubic',
                'cubic_packet_log.csv')
            trace_file = os.path.join(SAVE_DIR, 'rand_{}'.format(metric),
                                    str(val), 'config_{}'.format(config_id),
                                    'trace_{}'.format(i), 'trace.json')


            udr_big_sim_log_file = os.path.join(
                SAVE_DIR, 'rand_{}'.format(metric), str(val),
                'config_{}'.format(config_id), 'trace_{}'.format(i), 'udr_big',
                'aurora_simulation_log.csv')
            udr_big_pkt_log_file = os.path.join(
                SAVE_DIR, 'rand_{}'.format(metric), str(val),
                'config_{}'.format(config_id), 'trace_{}'.format(i), 'udr_big',
                'aurora_packet_log.csv')

            udr_mid_sim_log_file = os.path.join(
                SAVE_DIR, 'rand_{}'.format(metric), str(val),
                'config_{}'.format(config_id), 'trace_{}'.format(i), 'udr_mid',
                'aurora_simulation_log.csv')
            udr_mid_pkt_log_file = os.path.join(
                SAVE_DIR, 'rand_{}'.format(metric), str(val),
                'config_{}'.format(config_id), 'trace_{}'.format(i), 'udr_mid',
                'aurora_packet_log.csv')

            udr_small_sim_log_file = os.path.join(
                SAVE_DIR, 'rand_{}'.format(metric), str(val),
                'config_{}'.format(config_id), 'trace_{}'.format(i), 'udr_small',
                'aurora_simulation_log.csv')
            udr_small_pkt_log_file = os.path.join(
                SAVE_DIR, 'rand_{}'.format(metric), str(val),
                'config_{}'.format(config_id), 'trace_{}'.format(i), 'udr_small',
                'aurora_packet_log.csv')

            bo_sim_log_file = os.path.join(
                SAVE_DIR, 'rand_{}'.format(metric), str(val),
                'config_{}'.format(config_id), 'trace_{}'.format(i), 'bo',
                'aurora_simulation_log.csv')
            bo_pkt_log_file = os.path.join(
                SAVE_DIR, 'rand_{}'.format(metric), str(val),
                'config_{}'.format(config_id), 'trace_{}'.format(i), 'bo',
                'aurora_packet_log.csv')

            # if i == 0:
            #     for cc_name, sim_log_file, pkt_log_file in zip(
            #             ['cubic', 'bo'],
            #             [cubic_sim_log_file, udr_small_sim_log_file,
            #              udr_mid_sim_log_file, udr_big_sim_log_file, bo_sim_log_file],
            #             [cubic_pkt_log_file, udr_small_pkt_log_file,
            #              udr_mid_pkt_log_file, udr_big_pkt_log_file, bo_pkt_log_file]):
            #
            #         cmd = "python ../plot_scripts/plot_packet_log.py --log-file {} " \
            #             "--save-dir {} --trace-file {}".format(
            #                 pkt_log_file,
            #                 os.path.join(SAVE_DIR, 'rand_{}'.format(metric), str(val),
            #                              'config_{}'.format(config_id), 'trace_{}'.format(i), cc_name),
            #                 trace_file)
            #         subprocess.check_output(cmd, shell=True).strip()
            #         cmd = "python ../plot_scripts/plot_time_series.py --log-file {} " \
            #             "--save-dir {} --trace-file {}".format(
            #                 sim_log_file,
            #                 os.path.join(SAVE_DIR, 'rand_{}'.format(metric), str(val),
            #                              'config_{}'.format(config_id),
            #                              'trace_{}'.format(i), cc_name), trace_file)
            #         print(cmd)
            #         subprocess.check_output(cmd, shell=True).strip()

            bo_sim_log = pd.read_csv(bo_sim_log_file)
            # trace_bo_rewards.append(bo_sim_log['reward'].mean())
            trace_bo_rewards.append(
            PacketLog.from_log_file(bo_pkt_log_file).get_reward(trace_file))

            udr_big_sim_log = pd.read_csv(udr_big_sim_log_file)
            # trace_udr_big_rewards.append(udr_big_sim_log['reward'].mean())
            trace_udr_big_rewards.append(
            PacketLog.from_log_file(udr_big_pkt_log_file).get_reward(trace_file))

            udr_mid_sim_log = pd.read_csv(udr_mid_sim_log_file)
            # trace_udr_mid_rewards.append(udr_mid_sim_log['reward'].mean())
            trace_udr_mid_rewards.append(
            PacketLog.from_log_file(udr_mid_pkt_log_file).get_reward(trace_file))

            udr_small_sim_log = pd.read_csv(udr_small_sim_log_file)
            # trace_udr_small_rewards.append(udr_small_sim_log['reward'].mean())
            trace_udr_small_rewards.append(
            PacketLog.from_log_file(udr_small_pkt_log_file).get_reward(trace_file))

            cubic_sim_log = pd.read_csv(cubic_sim_log_file)
            trace_cubic_rewards.append(cubic_sim_log['reward'].mean())
            trace_cubic_rewards.append(
            PacketLog.from_log_file(cubic_pkt_log_file).get_reward(trace_file))

        config_bo_avg_rewards.append(np.mean(trace_bo_rewards))
        config_bo_avg_rewards_errs.append(
            np.std(trace_bo_rewards)/ np.sqrt(len(trace_bo_rewards)))

        config_udr_big_avg_rewards.append(np.mean(trace_udr_big_rewards))
        config_udr_big_avg_rewards_errs.append(
            np.std(trace_udr_big_rewards) / np.sqrt(len(trace_udr_big_rewards)))

        config_udr_mid_avg_rewards.append(np.mean(trace_udr_mid_rewards))
        config_udr_mid_avg_rewards_errs.append(
            np.std(trace_udr_mid_rewards) / np.sqrt(len(trace_udr_mid_rewards)))

        config_udr_small_avg_rewards.append(np.mean(trace_udr_small_rewards))
        config_udr_small_avg_rewards_errs.append(
            np.std(trace_udr_small_rewards) / np.sqrt(len(trace_udr_small_rewards)))

        config_cubic_avg_rewards.append(np.mean(trace_cubic_rewards))
        config_cubic_avg_rewards_errs.append(
            np.std(trace_cubic_rewards) / np.sqrt(len(trace_cubic_rewards)))

    plt.figure()
    plt.errorbar(vals2test[metric], config_cubic_avg_rewards,
                 yerr=config_cubic_avg_rewards_errs,
                 marker='o', linestyle='-', c="C0", label='TCP Cubic')

    plt.errorbar(vals2test[metric], config_udr_big_avg_rewards,
                 yerr=config_udr_big_avg_rewards_errs, marker='o',
                 linestyle='-', c="C1", label='UDR Big')

    plt.errorbar(vals2test[metric], config_udr_mid_avg_rewards,
                 yerr=config_udr_mid_avg_rewards_errs, marker='1',
                 linestyle='-', c="C1", label='UDR Mid')

    plt.errorbar(vals2test[metric], config_udr_small_avg_rewards,
                 yerr=config_udr_small_avg_rewards_errs, marker='+',
                 linestyle='-', c="C1", label='UDR Small')

    plt.errorbar(vals2test[metric], config_bo_avg_rewards,
                 yerr=config_bo_avg_rewards_errs, marker='s',
                 linestyle='--', c="C1", label='Bo')

    # plt.errorbar(vals, bo_rewards, yerr=bo_rewards_errs, marker='o',
    #              linestyle='--', c="C1", label='BO')
    plt.xlabel(metric)

    plt.title('default: bw={:.1f}, delay={:.1f}, loss={:.1f}, queue={:.1f}, T_s={:.1f}, delay_noise={:.1f}'.format(
                 bandwidth, delay, loss, queue, T_s, delay_noise))
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'rand_{}'.format(metric), 'config_{}_with_bo.png'.format(config_id)))
    plt.close()

    avg_bo_rewards.append(config_bo_avg_rewards)
    avg_bo_rewards_errs.append(config_bo_avg_rewards_errs)

    avg_udr_big_rewards.append(config_udr_big_avg_rewards)
    avg_udr_big_rewards_errs.append(config_udr_big_avg_rewards_errs)

    avg_udr_mid_rewards.append(config_udr_mid_avg_rewards)
    avg_udr_mid_rewards_errs.append(config_udr_mid_avg_rewards_errs)
    avg_udr_small_rewards.append(config_udr_small_avg_rewards)
    avg_udr_small_rewards_errs.append(config_udr_small_avg_rewards_errs)

    avg_cubic_rewards.append(config_cubic_avg_rewards)
    avg_cubic_rewards_errs.append(config_cubic_avg_rewards_errs)

avg_cubic_rewards = np.mean(avg_cubic_rewards, axis=0)
avg_cubic_rewards_errs = np.mean(avg_cubic_rewards_errs, axis=0)

avg_udr_big_rewards = np.mean(avg_udr_big_rewards, axis=0)
avg_udr_big_rewards_errs = np.mean(avg_udr_big_rewards_errs, axis=0)

avg_udr_mid_rewards = np.mean(avg_udr_mid_rewards, axis=0)
avg_udr_mid_rewards_errs = np.mean(avg_udr_mid_rewards_errs, axis=0)


avg_udr_small_rewards = np.mean(avg_udr_small_rewards, axis=0)
avg_udr_small_rewards_errs = np.mean(avg_udr_small_rewards_errs, axis=0)

avg_bo_rewards = np.mean(avg_bo_rewards, axis=0)
avg_bo_rewards_errs = np.mean(avg_bo_rewards_errs, axis=0)

if metric == 'delay':
    vals2test[metric] = [val * 2 for val in vals2test[metric]]
elif metric == 'bandwidth':
    vals2test[metric] = [val + 1 for val in vals2test[metric]]


with open(os.path.join(SAVE_DIR, 'rand_{}'.format(metric), 'eval_in_sim_{}.csv'.format(metric)), 'w', 1) as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['vals', 'avg_cubic_rewards', 'cubic_rewards_errs',
                     'avg_udr_big_rewards', 'udr_big_rewards_errs',
                     'avg_udr_mid_rewards', 'udr_mid_rewards_errs',
                     'avg_udr_small_rewards', 'udr_small_rewards_errs', 'avg_bo_rewards', 'bo_rewards_errs'])
    try:
        writer.writerows(zip(vals2test[metric], avg_cubic_rewards,
                         avg_cubic_rewards_errs,
                         avg_udr_big_rewards, avg_cubic_rewards_errs,
                         avg_udr_mid_rewards, avg_udr_mid_rewards_errs,
                         avg_udr_small_rewards, avg_udr_small_rewards_errs,
                         avg_bo_rewards, avg_bo_rewards_errs))
    except:
        import ipdb
        ipdb.set_trace()

vals = []
avg_cubic_rewards = []
cubic_low_bnd = []
cubic_up_bnd = []

avg_udr_big_rewards = []
udr_big_low_bnd = []
udr_big_up_bnd = []

avg_udr_mid_rewards = []
udr_mid_low_bnd = []
udr_mid_up_bnd = []

avg_udr_small_rewards = []
udr_small_low_bnd = []
udr_small_up_bnd = []

avg_bo_rewards = []
bo_low_bnd = []
bo_up_bnd = []
with open(os.path.join(SAVE_DIR, 'rand_{}'.format(metric), 'eval_in_sim_{}.csv'.format(metric)), 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        vals.append(float(row['vals']))
        avg_cubic_rewards.append(float(row['avg_cubic_rewards']))
        cubic_low_bnd.append(float(row['avg_cubic_rewards']) - float(row['cubic_rewards_errs']))
        cubic_up_bnd.append(float(row['avg_cubic_rewards']) + float(row['cubic_rewards_errs']))

        avg_udr_big_rewards.append(float(row['avg_udr_big_rewards']))
        udr_big_low_bnd.append(float(row['avg_udr_big_rewards']) - float(row['udr_big_rewards_errs']))
        udr_big_up_bnd.append(float(row['avg_udr_big_rewards']) + float(row['udr_big_rewards_errs']))

        avg_udr_mid_rewards.append(float(row['avg_udr_mid_rewards']))
        udr_mid_low_bnd.append(float(row['avg_udr_mid_rewards']) - float(row['udr_mid_rewards_errs']))
        udr_mid_up_bnd.append(float(row['avg_udr_mid_rewards']) + float(row['udr_mid_rewards_errs']))

        avg_udr_small_rewards.append(float(row['avg_udr_small_rewards']))
        udr_small_low_bnd.append(float(row['avg_udr_small_rewards']) - float(row['udr_small_rewards_errs']))
        udr_small_up_bnd.append(float(row['avg_udr_small_rewards']) + float(row['udr_small_rewards_errs']))

        avg_bo_rewards.append(float(row['avg_bo_rewards']))
        bo_low_bnd.append(float(row['avg_bo_rewards']) - float(row['bo_rewards_errs']))
        bo_up_bnd.append(float(row['avg_bo_rewards']) + float(row['bo_rewards_errs']))

fig, ax = plt.subplots()


ax.plot(vals, avg_bo_rewards, color='blue', linewidth=2, linestyle='-', label="DRL+GDR")
ax.fill_between(vals, bo_low_bnd, bo_up_bnd, color='blue', alpha=0.1)

ax.plot(vals, avg_cubic_rewards , color='black',
        linestyle=':', linewidth=2.5, alpha=0.8, label="TCP Cubic")
ax.fill_between(vals, cubic_low_bnd, cubic_up_bnd, color='black' ,alpha=0.1)

ax.plot(vals, avg_udr_small_rewards, color='green', linewidth=3, linestyle=':', label="DRL+UDR_1")
ax.fill_between(vals, udr_small_low_bnd, udr_small_up_bnd, color='green', alpha=0.2)

ax.plot(vals, avg_udr_mid_rewards, color='green', linewidth=2,linestyle='--', label="DRL+UDR_2")
ax.fill_between(vals, udr_mid_low_bnd, udr_mid_up_bnd, color='green', alpha=0.1)

ax.plot(vals, avg_udr_big_rewards, color='green', linewidth=2, linestyle='-.', label="DRL+UDR_3")
ax.fill_between(vals, udr_big_low_bnd, udr_big_up_bnd, color='green', alpha=0.1)

# # plt.errorbar(vals2test[metric], np.mean(avg_cubic_rewards, axis=0),
# #              yerr=np.mean(avg_cubic_rewards_errs, axis=0),
# #              marker='o', linestyle='-', c="C0", label='TCP Cubic')
# # plt.errorbar(vals2test[metric], np.mean(avg_udr_big_rewards, axis=0),
# #              yerr=np.mean(avg_udr_big_rewards_errs, axis=0),
# #              linestyle='-', c="C1", label='UDR Big')
# # plt.errorbar(vals2test[metric], np.mean(avg_bo_rewards, axis=0),
# #              yerr=np.mean(avg_bo_rewards_errs, axis=0),
# #              linestyle='--', c="C1", label='Bo')
if metric == 'bandwidth':
    ax.set_xlabel('Max bandwidth (Mbps)')
elif metric == 'delay':
    ax.set_xlabel('Link RTT (ms)')
elif metric == 'T_s':
    ax.set_xlabel('Bandwidth Changing Frequency (s)')

ax.legend()
ax.set_ylabel('Test performance')
# plt.savefig(os.path.join(SAVE_DIR, 'rand_{}'.format(metric), 'config_avg_with_bo.png'))
plt.savefig(os.path.join(SAVE_DIR, 'rand_{}'.format(metric), 'eval_in_sim_{}.pdf'.format(metric)), bbox_inches='tight')
plt.close()
