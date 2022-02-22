import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_ROOT = '../../figs_sigcomm22'

plt.style.use('seaborn-deep')
plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = 42
# plt.rcParams['axes.labelsize'] = 42
# plt.rcParams['legend.fontsize'] = 42
# plt.rcParams['figure.figsize'] = (11, 9)
plt.rcParams['font.size'] = 36
plt.rcParams['axes.labelsize'] = 36
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['legend.fontsize'] = 36
plt.rcParams['svg.fonttype'] = 'none'

HATCHES = ['/', '\\', 'x', 'o', '.', 'O', '-', '*', '+']
WIDTH = 0.3

bbr_reward, bbr_tput, bbr_tail_lat, bbr_loss = 118.07, 5.23, 517.02, 0.05
copa_reward, copa_tput, copa_tail_lat, copa_loss = 255.84, 4.58, 333.47, 0.01
cubic_reward, cubic_tput, cubic_tail_lat, cubic_loss = 69.75, 5.40, 858.46, 0.02
vivace_reward, vivace_tput, vivace_tail_lat, vivace_loss = -404.59, 4.04, 864.41, 0.21
vivace_latency_reward, vivace_latency_tput, vivace_latency_tail_lat, vivace_latency_loss = -422.16, 4.40, 888.76, 0.22
vivace_loss_reward = -616.31 #5.04    941.72  0.32


genet_reward = 252.28
genet_reward_err = 6.46
genet_tput, genet_tail_lat, genet_loss = 5.02, 251.02, 0.02
udr1_reward = 142.31
udr1_reward_err = 23.78     #
udr1_tput, udr1_tail_lat, udr1_loss = 4.59, 418.87, 0.03
udr2_reward = 187.61
udr2_reward_err = 5.03     #
udr2_tput, udr2_tail_lat, udr2_loss = 4.74, 408.95, 0.01
udr3_reward = 203.96
udr3_reward_err = 4.05     # 4.74    386.01  0.01
udr3_tput, udr3_tail_lat, udr3_loss = 4.74, 386.01, 0.01
real_reward = 171.61
real_reward_err = 3.18      # 5.01    459.23  0.02
cl1_reward = 206.56
cl1_reward_err = 3.07        # 4.88    413.40  0.01
cl2_reward = 211.89
cl2_reward_err = 4.05      # 4.82    419.74  0.00

column_wid = 0.7
capsize_wid = 8
eline_wid = 2

def cellular_bars():
    plt.figure(figsize=(9,5))
    ax = plt.gca()
    # plt.bar([1, 2], [bbr_reward, cubic_reward])
    bars = plt.bar([1, 2, 3, 4],
        [udr1_reward, udr2_reward, udr3_reward, real_reward],
        yerr=[udr1_reward_err, udr2_reward_err, udr3_reward_err, real_reward_err],
        color='C0', width=column_wid, error_kw=dict( lw=eline_wid, capsize=capsize_wid))
    for bar, pat in zip(bars, HATCHES):
        bar.set_hatch(pat)

    plt.bar([5], [genet_reward], yerr=[genet_reward_err], color='C2',
            width=column_wid, error_kw=dict( lw=eline_wid, capsize=capsize_wid))
    # plt.title('Ethernet')
    ax = plt.gca()
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
    svg_file = os.path.join(SAVE_ROOT, 'evaluation_generalization_test_cellular.svg')
    pdf_file = os.path.join(SAVE_ROOT, 'evaluation_generalization_test_cellular.pdf')
    plt.savefig(svg_file,  bbox_inches='tight')
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
    ax.annotate('BBR', (bbr_tail_lat+102, bbr_tput+0.01))
    ax.scatter([copa_tail_lat], [copa_tput], marker='>', s=msize, color='C1',
               label='Copa')
    ax.annotate('Copa', (copa_tail_lat, copa_tput+0.01))
    ax.scatter([cubic_tail_lat], [cubic_tput], marker='v', s=msize,
               color='darkorange', label='Cubic')
    ax.annotate('Cubic', (cubic_tail_lat+50, cubic_tput-0.12))
    ax.scatter([vivace_latency_tail_lat], [vivace_latency_tput], marker='^',
               s=msize, color='C3', label='Vivace')
    ax.annotate('Vivace', (vivace_latency_tail_lat, vivace_latency_tput))
    ax.scatter([udr1_tail_lat], [udr1_tput], marker='<', s=msize, color='C4',
               label='RL1')
    ax.annotate('RL1', (udr1_tail_lat+27, udr1_tput-0.13))
    ax.scatter([udr2_tail_lat], [udr2_tput], marker='p', s=msize, color='C5',
               label='RL2')
    ax.annotate('RL2', (udr2_tail_lat+100, udr2_tput+0.02))
    ax.scatter([udr3_tail_lat], [udr3_tput], marker='s', s=msize,
               color='indigo', label='RL3')
    ax.annotate('RL3', (udr3_tail_lat-13, udr3_tput+0.02))
    ax.scatter([genet_tail_lat], [genet_tput], s=msize, color='C2',
               label='Genet')
    ax.annotate('Genet', (genet_tail_lat+60, genet_tput+0.05))
    ax.set_ylabel('Throughput (Mbps)')
    ax.set_xlabel('90th percentile latency (ms)')
    ax.invert_xaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # fig.legend(bbox_to_anchor=(0, 1.02, 1, 0.14), ncol=4, loc="upper center",
    #             borderaxespad=0, borderpad=0.2, columnspacing=0.01, handletextpad=0.001)
    #
    svg_file = os.path.join(SAVE_ROOT, 'evaluation_cc_scatter_cellular.svg')
    pdf_file = os.path.join(SAVE_ROOT, 'evaluation_cc_scatter_cellular.pdf')
    fig.savefig(svg_file,  bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))

if __name__ == '__main__':
    cellular_bars()
    # cc_scatter()
