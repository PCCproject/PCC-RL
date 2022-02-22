import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common.utils import compute_std_of_mean

SAVE_ROOT = '../../figs_sigcomm22'

plt.style.use('seaborn-deep')
plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = 42
# plt.rcParams['axes.labelsize'] = 42
# plt.rcParams['legend.fontsize'] = 42
# plt.rcParams['figure.figsize'] = (11, 9)
plt.rcParams['svg.fonttype'] = 'none'

HATCHES = ['/', '\\', 'x', 'o', '.', 'O', '-', '*', '+']
WIDTH = 0.3

bbr_reward, bbr_tput, bbr_tail_lat, bbr_loss = 192.81, 32.94, 368.93, 0.03
copa_reward, copa_tput, copa_tail_lat, copa_loss = 183.89, 25.70, 265.02, 0.01
cubic_reward, cubic_tput, cubic_tail_lat, cubic_loss = -19.16, 33.99, 802.69, 0.02
vivace_reward, vivace_tput, vivace_tail_lat, vivace_loss = -547.01, 21.71, 947.25, 0.13
vivace_latency_reward, vivace_latency_tput, vivace_latency_tail_lat, vivace_latency_loss = -548.64, 21.84, 1010.43, 0.13
vivace_loss_reward, vivace_loss_tput, vivace_loss_tail_lat, vivace_loss_loss = -825.15, 28.89, 1125.94, 0.26


genet_reward = 223.88
genet_reward_err = 8.05
genet_tput, genet_tail_lat, genet_loss = 31.77, 183.75, 0.02
udr1_reward = 136.81
udr1_reward_err = 23.61
udr1_tput, udr1_tail_lat, udr1_loss = 23.16, 204.23, 0.03
udr2_reward = 158.48
udr2_reward_err = 17.71
udr2_tput, udr2_tail_lat, udr2_loss = 23.09, 185.58, 0.02
udr3_reward = 159.34
udr3_reward_err = 22.83
udr3_tput, udr3_tail_lat, udr3_loss = 22.72, 179.06, 0.02
real_reward = 191.61
real_reward_err = 3.88      # 26.39   250.47  0.02
cl1_reward = 143.86
cl1_reward_err = 7.64      # 22.53   206.07  0.02
cl2_reward = 177.97
cl2_reward_err = 4.55      # 23.17   204.86  0.01


udr3_real_5percent_ethernet_rewards = [177.2, 209.8, 95.2]
udr3_real_10percent_ethernet_rewards = [139, 175, 173]
udr3_real_20percent_ethernet_rewards = [133, 125, 151]
udr3_real_50percent_ethernet_rewards = [162, 124, 78]


column_wid = 0.7
capsize_wid = 8
eline_wid = 2

def generalization_test_ethernet():
    plt.rcParams['font.size'] = 36
    plt.rcParams['axes.labelsize'] = 36
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['legend.fontsize'] = 36
    fig, ax = plt.subplots(figsize=(9, 5))
    # plt.bar([1, 2], [bbr_reward, cubic_reward], hatch=HATCHES[:2])
    bars = ax.bar([1, 2, 3, 4],
           [udr1_reward, udr2_reward, udr3_reward, real_reward],
           yerr=[udr1_reward_err, udr2_reward_err, udr3_reward_err,
                 real_reward_err], color='C0', width=column_wid,
           error_kw=dict( lw=eline_wid, capsize=capsize_wid))
    # bars = ax.bar([1, 2, 3, 4],
    #         [udr1_reward, udr2_reward, udr3_reward, real_reward],
    #         color=None, edgecolor='white')
    for bar, pat in zip(bars, HATCHES):
        bar.set_hatch(pat)

    ax.bar([5], [genet_reward], yerr=[genet_reward_err], capsize=8, width=column_wid,
           color='C2', error_kw=dict( lw=eline_wid, capsize=capsize_wid))
    # plt.title('Ethernet')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['RL1', 'RL2', 'RL3', 'RL-real', 'Genet'], rotation=20)
    ax.set_ylabel('Test reward')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)         # ticks along the top edge are off
    # plt.tight_layout()
    svg_file = os.path.join(SAVE_ROOT, 'evaluation_generalization_test_ethernet.svg')
    pdf_file = os.path.join(SAVE_ROOT, 'evaluation_generalization_test_ethernet.pdf')
    fig.savefig(svg_file,  bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))



def asymptotic_real():
    plt.rcParams['font.size'] = 34
    plt.rcParams['axes.labelsize'] = 34
    plt.rcParams['axes.titlesize'] = 34
    plt.rcParams['legend.fontsize'] = 34
    fig, ax = plt.subplots(figsize=(9.5, 5))
    bbr_reward = 192.81 #32.94   368.93  0.03
    udr_real_synthetic_reward =  171.16 #    23.67   194.00  0.02
    udr_real_synthetic_reward_err =  24.22
    genet_real_synthetic_reward = 239.39   # 30.93   208.04  0.02
    genet_real_synthetic_reward_err = 7.34
    cubic_reward = 97.16 #  33.99   802.69  0.02


    # plt.bar([1, 2], [bbr_reward, cubic_reward])
    # plt.bar([3.5, 4.5,],
    ax.bar([1, 2.2, 3.4, 4.6, 5.8],
            [udr_real_synthetic_reward, # 1%
             np.mean(udr3_real_10percent_ethernet_rewards),
             np.mean(udr3_real_20percent_ethernet_rewards),
             np.mean(udr3_real_50percent_ethernet_rewards),
             real_reward],
            yerr=[udr_real_synthetic_reward_err,
              compute_std_of_mean(udr3_real_10percent_ethernet_rewards),
              compute_std_of_mean(udr3_real_20percent_ethernet_rewards),
              compute_std_of_mean(udr3_real_50percent_ethernet_rewards),

                  real_reward_err], hatch=HATCHES[:5], capsize=8)

    ax.bar([7.3], [genet_real_synthetic_reward],
           yerr=[genet_real_synthetic_reward_err], capsize=8, color='C2')

    ax.set_xticks([1, 2.2, 3.4, 4.6, 5.8, 7.3])
    ax.set_xticklabels(['5%', '10%',
                        '20%', '50%',
                        '100%', 'Genet\n(synthetic+real)'],)
                        # rotation=30, ha='right', rotation_mode='anchor')
    ax.annotate("RL (synthetic + real)", (0.9, 209))
    ax.set_ylabel('Test reward')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)         # ticks along the top edge are off
    # plt.tight_layout()

    svg_file = os.path.join(SAVE_ROOT, 'evaluation_asymptotic_real_new.svg')
    pdf_file = os.path.join(SAVE_ROOT, 'evaluation_asymptotic_real_new.pdf')
    fig.savefig(svg_file,  bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))


def cc_scatter():
    plt.rcParams['font.size'] = 32
    plt.rcParams['axes.labelsize'] = 32
    plt.rcParams['axes.titlesize'] = 32
    plt.rcParams['legend.fontsize'] = 32
    fig, ax = plt.subplots(figsize=(9, 5))
    msize = 200
    ax.scatter([bbr_tail_lat], [bbr_tput], marker='d', s=msize, color='C0',
               label='BBR')
    ax.annotate('BBR', (bbr_tail_lat+70, bbr_tput-2))
    ax.scatter([copa_tail_lat], [copa_tput], marker='>', s=msize, color='C1',
               label='Copa')
    ax.annotate('Copa', (copa_tail_lat+110, copa_tput+0.8))
    ax.scatter([cubic_tail_lat], [cubic_tput], marker='v',
               s=msize, color='darkorange', label='Cubic')
    ax.annotate('Cubic', (cubic_tail_lat+80, cubic_tput - 2))
    ax.scatter([vivace_latency_tail_lat], [vivace_latency_tput], marker='^',
               s=msize, color='C3', label='Vivace')
    ax.annotate('Vivace', (vivace_latency_tail_lat, vivace_latency_tput))
    ax.scatter([udr1_tail_lat], [udr1_tput], marker='<', s=msize, color='C4',
               label='RL1')
    ax.annotate('RL1', (udr1_tail_lat+115, udr1_tput))
    ax.scatter([udr2_tail_lat], [udr2_tput], marker='p', s=msize, color='C5',
               label='RL2')
    ax.annotate('RL2', (udr2_tail_lat+20, udr2_tput+0.5))
    ax.scatter([udr3_tail_lat], [udr3_tput], marker='s', s=msize, color='indigo', label='RL3')
    ax.annotate('RL3', (udr3_tail_lat+120, udr3_tput-1.2))
    ax.scatter([genet_tail_lat], [genet_tput], s=msize, color='C2',
               label='Genet')
    ax.annotate('Genet', (genet_tail_lat+80, genet_tput-1.5))
    ax.set_ylabel('Throughput (Mbps)')
    ax.set_xlabel('90th percentile latency (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_xaxis()
    # fig.legend(bbox_to_anchor=(0, 1.02, 1, 0.14), ncol=4, loc="upper center",
    #            borderaxespad=0, borderpad=0.2, columnspacing=0.01,
    #            handletextpad=0.001)

    svg_file = os.path.join(SAVE_ROOT, 'evaluation_cc_scatter.svg')
    pdf_file = os.path.join(SAVE_ROOT, 'evaluation_cc_scatter.pdf')
    fig.savefig(svg_file,  bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))



if __name__ == '__main__':
    # generalization_test_ethernet()
    asymptotic_real()
    # cc_scatter()
