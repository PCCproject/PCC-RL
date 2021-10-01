import csv
import os
import subprocess

from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from common.utils import read_json_file
from plot_scripts.plot_packet_log import PacketLog
from simulator.aurora import Aurora
from simulator.network_simulator.cubic import Cubic
from simulator.network_simulator.bbr import BBR
# from simulator.cubic import Cubic
from simulator.trace import generate_trace

UDR_SMALL_MODEL_PATH = "../../models/udr_small_lossless/seed_20/model_step_187200.ckpt"
UDR_MID_MODEL_PATH = "../../models/udr_mid_lossless/seed_20/model_step_86400.ckpt"
UDR_LARGE_MODEL_PATH = "../../models/udr_large_lossless/seed_20/model_step_482400.ckpt"
# GENET_MODEL_PATH = "../../models/udr_large_lossless/seed_20/model_step_2124000.ckpt"
# GENET_MODEL_PATH = "test/bo_3/model_step_72000.ckpt"
# GENET_MODEL_PATH = "test/bo_3/model_step_72000.ckpt"
GENET_MODEL_PATH = "test/bo_3/model_step_72000.ckpt"
GENET_MODEL_PATH = "../../models/genet_bbr_no_noise_udr_large_start/bo_9/model_step_7200.ckpt"
UDR_TRAIN_CONFIG_FILE = "../../config/train/udr_7_dims_0515/udr_large_lossless.json"
DIMS_TO_VARY = ['delay', 'loss', 'queue', 'T_s']
# DIMS_TO_VARY = ['T_s', ]
DIMS_TO_VARY = ['bandwidth']
# DIM_UNITS = ['ms', '', 'packets', 's']
DIM_UNITS = ['Mbps', ]
RESULT_ROOT = "../../results"
EXP_NAME1 = "sim_eval_bbr_reproduce"
EXP_NAME = "sim_eval_reproduce"

plt.style.use('seaborn-deep')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 65
plt.rcParams['axes.labelsize'] = 65
plt.rcParams['legend.fontsize'] = 40
plt.rcParams['legend.columnspacing'] = 0.5
plt.rcParams['legend.labelspacing'] = 0.002
plt.rcParams['figure.figsize'] = (11, 9)


# DEFAULT_CONFIGS = []
DEFAULT_VALUES = {
    "bandwidth": 10,
    'delay': 25,
    "loss": 0,
    "queue": 20,
    "T_s": 3,
    "delay_noise": 0
}


def main():
    bbr = BBR(True)
    cubic = Cubic(True)
    # cubic = Cubic(20)
    genet = Aurora(seed=20, log_dir=RESULT_ROOT,
                   pretrained_model_path=GENET_MODEL_PATH,
                   timesteps_per_actorbatch=10, delta_scale=1)
    udr_small = Aurora(seed=20, log_dir=RESULT_ROOT,
                       pretrained_model_path=UDR_SMALL_MODEL_PATH,
                       timesteps_per_actorbatch=10, delta_scale=1)
    udr_mid = Aurora(seed=20, log_dir=RESULT_ROOT,
                     pretrained_model_path=UDR_MID_MODEL_PATH,
                     timesteps_per_actorbatch=10, delta_scale=1)
    udr_large = Aurora(seed=20, log_dir=RESULT_ROOT,
                       pretrained_model_path=UDR_LARGE_MODEL_PATH,
                       timesteps_per_actorbatch=10, delta_scale=1)

    # for _ in range(10):
    #     DEFAULT_CONFIGS.append(
    #         {
    #          "bandwidth": 10 ** np.random.uniform(np.log10(1), np.log10(10), 1).item(),
    #          "delay": np.random.uniform(5, 200, 1).item(),
    #          "loss": np.random.uniform(0, 0.0, 1).item(),
    #          "queue": 10 ** np.random.uniform(np.log10(2), np.log10(30), 1).item(),
    #          "T_s": np.random.randint(0, 6, 1).item(),
    #          "delay_noise": np.random.uniform(0, 0, 1).item()})
    # print(DEFAULT_CONFIGS)
    udr_train_config = read_json_file(UDR_TRAIN_CONFIG_FILE)[0]
    # print(udr_train_config)
    for dim, unit in zip(DIMS_TO_VARY, DIM_UNITS):
        print(dim, udr_train_config[dim])
        if dim == 'bandwidth':
            vals_to_test = 10**np.linspace(
                np.log10(1), np.log10(udr_train_config[dim][1]), 10)
        elif dim == 'queue':
            vals_to_test = 10**np.linspace(
                np.log10(udr_train_config[dim][0]), np.log10(udr_train_config[dim][1]), 10)
        elif dim == 'loss':
            vals_to_test = np.linspace(0, 0.005, 10)
        else:
            vals_to_test = np.linspace(
                udr_train_config[dim][0], udr_train_config[dim][1], 10)
        print(vals_to_test)
        bbr_avg_rewards = []
        bbr_reward_errs = []
        cubic_avg_rewards = []
        cubic_reward_errs = []
        udr_small_avg_rewards = []
        udr_small_reward_errs = []
        udr_mid_avg_rewards = []
        udr_mid_reward_errs = []
        udr_large_avg_rewards = []
        udr_large_reward_errs = []
        genet_avg_rewards = []
        genet_reward_errs = []
        for val_idx, val in enumerate(vals_to_test):
            if dim == 'bandwidth':
                max_bw = val
            else:
                max_bw = DEFAULT_VALUES['bandwidth']

            if dim == 'delay':
                min_delay, max_delay = val, val
            else:
                min_delay, max_delay = DEFAULT_VALUES['delay'], DEFAULT_VALUES['delay']
            if dim == 'loss':
                min_loss, max_loss = val, val
            else:
                min_loss, max_loss = DEFAULT_VALUES['loss'], DEFAULT_VALUES['loss']
            if dim == 'queue':
                min_queue, max_queue = val, val
            else:
                min_queue, max_queue = DEFAULT_VALUES['queue'], DEFAULT_VALUES['queue']
            if dim == 'T_s':
                min_T_s, max_T_s = val, val
            else:
                min_T_s, max_T_s = DEFAULT_VALUES['T_s'], DEFAULT_VALUES['T_s']

            # generate n=10 traces for each config
            traces = [generate_trace(duration_range=(30, 30),
                                     bandwidth_range=(1, max_bw),
                                     delay_range=(min_delay, max_delay),
                                     loss_rate_range=(min_loss, max_loss),
                                     queue_size_range=(min_queue, max_queue),
                                     T_s_range=(min_T_s, max_T_s),
                                     delay_noise_range=(0, 0),
                                     constant_bw=False, seed=i) for i in range(10)]
            bbr_rewards = []
            cubic_rewards = []
            udr_small_rewards = []
            udr_mid_rewards = []
            udr_large_rewards = []
            genet_rewards = []
            for i, trace in enumerate(tqdm(traces)):
                save_dir = os.path.join(RESULT_ROOT, EXP_NAME, "vary_{}".format(
                    dim), "val_{}".format(val_idx), "trace_{}".format(i))
                os.makedirs(save_dir, exist_ok=True)
                trace_file = os.path.join(save_dir, "trace_{}.json".format(i))
                trace.dump(trace_file)

                # bbr
                # save_dir = os.path.join(RESULT_ROOT, EXP_NAME, "vary_{}".format(
                #     dim), "val_{}".format(val_idx), "trace_{}".format(i), "bbr")
                # os.makedirs(save_dir, exist_ok=True)
                #
                # if os.path.exists(os.path.join(save_dir, "bbr_packet_log.csv")):
                #     pkt_log = PacketLog.from_log_file(
                #         os.path.join(save_dir, "bbr_packet_log.csv"))
                #     pkt_level_reward = pkt_log.get_reward("", trace)
                # else:
                #     test_reward, pkt_level_reward = bbr.test(trace, save_dir)
                #
                # # bbr_rewards.append(test_reward)
                # bbr_rewards.append(pkt_level_reward)

                # cubic
                # save_dir = os.path.join(RESULT_ROOT, EXP_NAME, "vary_{}".format(
                #     dim), "val_{}".format(val_idx), "trace_{}".format(i), "cubic")
                # os.makedirs(save_dir, exist_ok=True)
                #
                # if os.path.exists(os.path.join(save_dir, "cubic_packet_log.csv")):
                #     pkt_log = PacketLog.from_log_file(
                #         os.path.join(save_dir, "cubic_packet_log.csv"))
                #     pkt_level_reward = pkt_log.get_reward("", trace)
                # else:
                #     test_reward, pkt_level_reward = cubic.test(trace, save_dir)
                # # cubic_rewards.append(test_reward)
                # cubic_rewards.append(pkt_level_reward)

                # cmd = "python ../plot_scripts/plot_packet_log.py --log-file {} " \
                #     "--save-dir {} --trace-file {}".format(
                #         os.path.join(save_dir, "cubic_packet_log.csv"),
                #         save_dir, trace_file)
                # subprocess.check_output(cmd, shell=True).strip()
                # cmd = "python ../plot_scripts/plot_time_series.py --log-file {} " \
                #     "--save-dir {} --trace-file {}".format(
                #         os.path.join(save_dir, "cubic_simulation_log.csv"),
                #         save_dir, trace_file)
                # subprocess.check_output(cmd, shell=True).strip()

                # genet
                save_dir = os.path.join(RESULT_ROOT, EXP_NAME, "vary_{}".format(
                    dim), "val_{}".format(val_idx), "trace_{}".format(i), "genet")
                if os.path.exists(os.path.join(save_dir, "aurora_packet_log.csv")):
                    pkt_log = PacketLog.from_log_file(
                        os.path.join(save_dir, "aurora_packet_log.csv"))
                else:
                    _, reward_list, _, _, _, _, _, _, _, pkt_log = genet.test(
                        trace, save_dir)
                    pkt_log = PacketLog.from_log(pkt_log)
                genet_rewards.append(pkt_log.get_reward("", trace))

                # udr_small
                # save_dir = os.path.join(RESULT_ROOT, EXP_NAME, "vary_{}".format(
                #     dim), "val_{}".format(val_idx), "trace_{}".format(i), "udr_small")
                # if os.path.exists(os.path.join(save_dir, "aurora_packet_log.csv")):
                #     pkt_log = PacketLog.from_log_file(
                #         os.path.join(save_dir, "aurora_packet_log.csv"))
                # else:
                #     _, reward_list, _, _, _, _, _, _, _, pkt_log = udr_small.test(
                #         trace, save_dir)
                #     pkt_log = PacketLog.from_log(pkt_log)
                # udr_small_rewards.append(pkt_log.get_reward("", trace))
                #
                # # udr_mid
                # save_dir = os.path.join(RESULT_ROOT, EXP_NAME, "vary_{}".format(
                #     dim), "val_{}".format(val_idx), "trace_{}".format(i), "udr_mid")
                # if os.path.exists(os.path.join(save_dir, "aurora_packet_log.csv")):
                #     pkt_log = PacketLog.from_log_file(
                #         os.path.join(save_dir, "aurora_packet_log.csv"))
                # else:
                #     _, reward_list, _, _, _, _, _, _, _, pkt_log = udr_mid.test(
                #         trace, save_dir)
                #     pkt_log = PacketLog.from_log(pkt_log)
                # udr_mid_rewards.append(pkt_log.get_reward("", trace))
                # _, reward_list, _, _, _, _, _, _, _, pkt_log = udr_mid.test(
                #     trace, save_dir)
                # # test_reward = np.mean(reward_list)
                # # udr_mid_rewards.append(test_reward)
                #
                # # udr_large
                # save_dir = os.path.join(RESULT_ROOT, EXP_NAME, "vary_{}".format(
                #     dim), "val_{}".format(val_idx), "trace_{}".format(i), "udr_large")
                # os.makedirs(save_dir, exist_ok=True)
                # if os.path.exists(os.path.join(save_dir, "aurora_packet_log.csv")):
                #     pkt_log = PacketLog.from_log_file(
                #         os.path.join(save_dir, "aurora_packet_log.csv"))
                # else:
                #     _, reward_list, _, _, _, _, _, _, _, pkt_log = udr_large.test(
                #         trace, save_dir)
                #     pkt_log = PacketLog.from_log(pkt_log)
                # # test_reward = np.mean(reward_list)
                # # udr_large_rewards.append(test_reward)
                # udr_large_rewards.append(pkt_log.get_reward("", trace))
                # # cmd = "python ../plot_scripts/plot_packet_log.py --log-file {} " \
                # #     "--save-dir {} --trace-file {}".format(
                # #         os.path.join(save_dir, "aurora_packet_log.csv"),
                # #         save_dir, trace_file)
                # # subprocess.check_output(cmd, shell=True).strip()
                # # cmd = "python ../plot_scripts/plot_time_series.py --log-file {} " \
                # #     "--save-dir {} --trace-file {}".format(
                # #         os.path.join(save_dir, "aurora_simulation_log.csv"),
                # #         save_dir, trace_file)
                # # subprocess.check_output(cmd, shell=True).strip()
                #
                # # # genet_model.test(trace)
            print(len(cubic_avg_rewards), len(udr_large_avg_rewards))
            bbr_avg_rewards.append(np.mean(bbr_rewards))
            bbr_reward_errs.append(
                np.std(bbr_rewards) / np.sqrt(len(bbr_rewards)))
            cubic_avg_rewards.append(np.mean(cubic_rewards))
            cubic_reward_errs.append(
                np.std(cubic_rewards) / np.sqrt(len(cubic_rewards)))
            udr_small_avg_rewards.append(np.mean(udr_small_rewards))
            udr_small_reward_errs.append(
                np.std(udr_small_rewards) / np.sqrt(len(udr_small_rewards)))
            udr_mid_avg_rewards.append(np.mean(udr_mid_rewards))
            udr_mid_reward_errs.append(
                np.std(udr_mid_rewards) / np.sqrt(len(udr_mid_rewards)))
            udr_large_avg_rewards.append(np.mean(udr_large_rewards))
            udr_large_reward_errs.append(
                np.std(udr_large_rewards) / np.sqrt(len(udr_large_rewards)))

            genet_avg_rewards.append(np.mean(genet_rewards))
            genet_reward_errs.append(
                np.std(genet_rewards) / np.sqrt(len(genet_rewards)))
        plt.figure()
        ax = plt.gca()

        import pdb
        pdb.set_trace()
        ax.plot(vals_to_test, genet_avg_rewards, color='C2',
                linewidth=4, alpha=1, linestyle='-', label="GENET")
        genet_low_bnd = np.array(genet_avg_rewards) - \
            np.array(genet_reward_errs)
        genet_up_bnd = np.array(genet_avg_rewards) + \
            np.array(genet_reward_errs)
        ax.fill_between(vals_to_test, genet_low_bnd,
                        genet_up_bnd, color='C2', alpha=0.1)

        ax.plot(vals_to_test, bbr_avg_rewards, color='C0',
                linestyle='-.', linewidth=4, alpha=1, label="BBR")
        bbr_low_bnd = np.array(bbr_avg_rewards) - \
            np.array(bbr_reward_errs)
        bbr_up_bnd = np.array(bbr_avg_rewards) + \
            np.array(bbr_reward_errs)
        ax.fill_between(vals_to_test, bbr_low_bnd,
                        bbr_up_bnd, color='C0', alpha=0.1)

        ax.plot(vals_to_test, cubic_avg_rewards, color='C0',
                linestyle='-', linewidth=4, alpha=1, label="TCP Cubic")
        cubic_low_bnd = np.array(cubic_avg_rewards) - \
            np.array(cubic_reward_errs)
        cubic_up_bnd = np.array(cubic_avg_rewards) + \
            np.array(cubic_reward_errs)
        ax.fill_between(vals_to_test, cubic_low_bnd,
                        cubic_up_bnd, color='C0', alpha=0.1)

        ax.plot(vals_to_test, udr_small_avg_rewards, color='grey',
                linewidth=4, linestyle=':', label="UDR-1")
        udr_small_low_bnd = np.array(
            udr_small_avg_rewards) - np.array(udr_small_reward_errs)
        udr_small_up_bnd = np.array(
            udr_small_avg_rewards) + np.array(udr_small_reward_errs)
        ax.fill_between(vals_to_test, udr_small_low_bnd,
                        udr_small_up_bnd, color='grey', alpha=0.1)

        ax.plot(vals_to_test, udr_mid_avg_rewards, color='grey',
                linewidth=4, linestyle='--', label="UDR-2")
        udr_mid_low_bnd = np.array(
            udr_mid_avg_rewards) - np.array(udr_mid_reward_errs)
        udr_mid_up_bnd = np.array(udr_mid_avg_rewards) + \
            np.array(udr_mid_reward_errs)
        ax.fill_between(vals_to_test, udr_mid_low_bnd,
                        udr_mid_up_bnd, color='grey', alpha=0.1)

        ax.plot(vals_to_test, udr_large_avg_rewards, color='grey',
                linewidth=4, linestyle='-.', label="UDR-3")
        udr_large_low_bnd = np.array(
            udr_large_avg_rewards) - np.array(udr_large_reward_errs)
        udr_large_up_bnd = np.array(
            udr_large_avg_rewards) + np.array(udr_large_reward_errs)
        ax.fill_between(vals_to_test, udr_large_low_bnd,
                        udr_large_up_bnd, color='grey', alpha=0.1)
        ax.set_xlabel("{}({})".format(dim, unit))
        ax.set_ylabel("Reward")
        ax.legend()
        plt.tight_layout()
        with open(os.path.join(RESULT_ROOT, EXP_NAME, "sim_eval_vary_{}_bbr_with_cubic.csv".format(dim)), 'w') as f:
            writer = csv.writer(f)
            writer.writerow([dim, 'genet_avg_rewards', 'genet_low_bnd', 'genet_up_bnd',
                'bbr_avg_rewards', 'bbr_low_bnd', 'bbr_up_bnd',
                'cubic_avg_rewards', 'cubic_low_bnd', 'cubic_up_bnd',
                'udr_small_avg_rewards', 'udr_small_low_bnd', 'udr_small_up_bnd',
                'udr_mid_avg_rewards', 'udr_mid_low_bnd', 'udr_mid_up_bnd',
                'udr_large_avg_rewards', 'udr_large_low_bnd', 'udr_large_up_bnd'])
            writer.writerows(zip(vals_to_test,
                    genet_avg_rewards, genet_low_bnd, genet_up_bnd,
                    bbr_avg_rewards, bbr_low_bnd, bbr_up_bnd,
                    cubic_avg_rewards, cubic_low_bnd, cubic_up_bnd,
                    udr_small_avg_rewards, udr_small_low_bnd, udr_small_up_bnd,
                    udr_mid_avg_rewards, udr_mid_low_bnd, udr_mid_up_bnd,
                    udr_large_avg_rewards, udr_large_low_bnd, udr_large_up_bnd))
        save_dir = os.path.join(RESULT_ROOT, EXP_NAME,
                                "sim_eval_vary_{}_bbr_with_cubic.png".format(dim))
        save_dir = os.path.join(RESULT_ROOT, EXP_NAME,
                                "sim_eval_vary_{}_bbr_with_cubic.pdf".format(dim))
        plt.savefig(save_dir)


if __name__ == "__main__":
    main()
