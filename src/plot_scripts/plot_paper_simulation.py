import pandas as pd
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 30
# plt.rcParams['legend.labelspacing'] = 0.02
# plt.rcParams['figure.figsize'] = (11, 9)

RESULT_ROOT = "../../results"

EXP_NAME = "sim_eval_reproduce"
# EXP_NAME1 = "sim_eval_bbr_reproduce"

fig, axes = plt.subplots(2, 2, figsize=(16, 8.5))
for dim, xlabel, ax in zip(['bandwidth', 'T_s', 'queue', 'loss'],
                           ['Bandwidth (Mbps)', 'Bandwidth change interval (s)',
                            'Queue (Packets)', 'Random packet loss rate (%)'], axes.flatten()):
    df = pd.read_csv(os.path.join(RESULT_ROOT, EXP_NAME, "sim_eval_vary_{}_bbr_with_cubic.csv".format(dim)))
    assert isinstance(df, pd.DataFrame)
    if dim == 'delay':
        vals_to_test = df[dim] * 2
    elif dim == 'loss':
        vals_to_test = df[dim] * 100
    else:
        vals_to_test = df[dim]
    genet_avg_rewards = df['genet_avg_rewards']
    genet_low_bnd = df['genet_low_bnd']
    genet_up_bnd = df['genet_up_bnd']
    bbr_avg_rewards = df['bbr_avg_rewards']
    bbr_low_bnd = df['bbr_low_bnd']
    bbr_up_bnd = df['bbr_up_bnd']
    cubic_avg_rewards = df['cubic_avg_rewards']
    cubic_low_bnd = df['cubic_low_bnd']
    cubic_up_bnd = df['cubic_up_bnd']
    udr_small_avg_rewards = df['udr_small_avg_rewards']
    udr_small_low_bnd = df['udr_small_low_bnd']
    udr_small_up_bnd = df['udr_small_up_bnd']
    udr_mid_avg_rewards = df['udr_mid_avg_rewards']
    udr_mid_low_bnd = df['udr_mid_low_bnd']
    udr_mid_up_bnd = df['udr_mid_up_bnd']
    udr_large_avg_rewards = df['udr_large_avg_rewards']
    udr_large_low_bnd = df['udr_large_low_bnd']
    udr_large_up_bnd = df['udr_large_up_bnd']

    ax.plot(vals_to_test, genet_avg_rewards, color='C2',
            linewidth=4, alpha=1, linestyle='-', label="GENET")
    # genet_low_bnd = np.array(genet_avg_rewards) - \
    #     np.array(genet_reward_errs)
    # genet_up_bnd = np.array(genet_avg_rewards) + \
    #     np.array(genet_reward_errs)
    ax.fill_between(vals_to_test, genet_low_bnd,
                    genet_up_bnd, color='C2', alpha=0.1)

    ax.plot(vals_to_test, bbr_avg_rewards, color='C0',
            linestyle='-', linewidth=4, alpha=1, label="BBR")
    # bbr_low_bnd = np.array(bbr_avg_rewards) - \
    #     np.array(bbr_reward_errs)
    # bbr_up_bnd = np.array(bbr_avg_rewards) + \
    #     np.array(bbr_reward_errs)
    ax.fill_between(vals_to_test, bbr_low_bnd, bbr_up_bnd, color='C0', alpha=0.1)

    ax.plot(vals_to_test, cubic_avg_rewards, color='C0',
            linestyle=':', linewidth=4, alpha=1, label="TCP Cubic")
    # cubic_low_bnd = np.array(cubic_avg_rewards) - \
    #     np.array(cubic_reward_errs)
    # cubic_up_bnd = np.array(cubic_avg_rewards) + \
    #     np.array(cubic_reward_errs)
    ax.fill_between(vals_to_test, cubic_low_bnd,
                    cubic_up_bnd, color='C0', alpha=0.1)

    ax.plot(vals_to_test, udr_small_avg_rewards, color='grey',
            linewidth=4, linestyle='-', label="UDR-1")
    # udr_small_low_bnd = np.array(
    #     udr_small_avg_rewards) - np.array(udr_small_reward_errs)
    # udr_small_up_bnd = np.array(
    #     udr_small_avg_rewards) + np.array(udr_small_reward_errs)
    ax.fill_between(vals_to_test, udr_small_low_bnd,
                    udr_small_up_bnd, color='grey', alpha=0.1)

    ax.plot(vals_to_test, udr_mid_avg_rewards, color='grey',
            linewidth=4, linestyle='--', label="UDR-2")
    # udr_mid_low_bnd = np.array(
    #     udr_mid_avg_rewards) - np.array(udr_mid_reward_errs)
    # udr_mid_up_bnd = np.array(udr_mid_avg_rewards) + \
    #     np.array(udr_mid_reward_errs)
    ax.fill_between(vals_to_test, udr_mid_low_bnd,
                    udr_mid_up_bnd, color='grey', alpha=0.1)

    ax.plot(vals_to_test, udr_large_avg_rewards, color='grey',
            linewidth=4, linestyle=':', label="UDR-3")
    # udr_large_low_bnd = np.array(
    #     udr_large_avg_rewards) - np.array(udr_large_reward_errs)
    # udr_large_up_bnd = np.array(
    #     udr_large_avg_rewards) + np.array(udr_large_reward_errs)
    ax.fill_between(vals_to_test, udr_large_low_bnd,
                    udr_large_up_bnd, color='grey', alpha=0.1)
    ax.set_xlim(min(vals_to_test), max(vals_to_test))
    ax.set_xlabel(xlabel)
    if dim == 'bandwidth' or dim == 'queue':
        ax.set_ylabel("Test Reward")
# plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
#                 mode="expand", borderaxespad=0, ncol=3)

line_labels = ["GENET", "BBR", "TCP Cubic", "UDR-1", "UDR-2", "UDR-3"]

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

fig.subplots_adjust(hspace=0.5)
fig.legend(handles=lines,     # The line objects
           labels=line_labels,   # The labels for each line
           loc="upper center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
            ncol=6, columnspacing=0.2, handletextpad=0.3,
            handlelength=1.2)
# plt.tight_layout()
    #     writer.writerow([dim, 'genet_avg_rewards', 'genet_low_bnd', 'genet_up_bnd',
    #         'bbr_avg_rewards', 'bbr_low_bnd', 'bbr_up_bnd',
    #         'cubic_avg_rewards', 'cubic_low_bnd', 'cubic_up_bnd',
    #         'udr_small_avg_rewards', 'udr_small_low_bnd', 'udr_small_up_bnd'
    #         'udr_mid_avg_rewards', 'udr_mid_low_bnd', 'udr_mid_up_bnd'
    #         'udr_large_avg_rewards', 'udr_large_low_bnd', 'udr_large_up_bnd'])
    #     writer.writerows(zip(vals_to_test,
    #             genet_avg_rewards, genet_low_bnd, genet_up_bnd,
    #             bbr_avg_rewards, bbr_low_bnd, bbr_up_bnd,
    #             cubic_avg_rewards, cubic_low_bnd, cubic_up_bnd,
    #             udr_small_avg_rewards, udr_small_low_bnd, udr_small_up_bnd,
    #             udr_mid_avg_rewards, udr_mid_low_bnd, udr_mid_up_bnd,
    #             udr_large_avg_rewards, udr_large_low_bnd, udr_large_up_bnd))
save_dir = os.path.join(RESULT_ROOT, EXP_NAME, "sim_eval_vary_summary.png")
save_dir = os.path.join(RESULT_ROOT, EXP_NAME, "sim_eval_vary_summary.pdf")
save_dir = os.path.join("../../figs/sim_eval_vary_summary.pdf")
plt.savefig(save_dir)
