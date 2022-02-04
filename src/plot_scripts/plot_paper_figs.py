import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, pearsonr

ROOT = '../../figs_sigcomm22'

plt.style.use('seaborn-deep')
plt.rcParams['font.family'] = 'Arial' #'Times New Roman' or
plt.rcParams['svg.fonttype'] = 'none'

HATCHES = ['/', '\\', 'x', '+', '-',  'o', 'O', '.', '*']

WIDTH = 0.3

def motivation_udr_baseline():
    # motivation figures
    plt.rcParams['font.size'] = 26
    plt.rcParams['axes.labelsize'] = 26
    plt.rcParams['axes.titlesize'] = 26
    plt.rcParams['legend.fontsize'] = 26
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    xvals = np.arange(0, 3) * (WIDTH + WIDTH / 3)
    data = pd.read_csv(os.path.join(ROOT, 'motivation_udr_baseline_asymptotic_perf.csv'))
    assert isinstance(data, pd.DataFrame)
    axes[0].bar(xvals,
           [data['udr1_gap'][0], data['udr2_gap'][0], data['udr3_gap'][0]],
            yerr=[data['udr1_gap_err'][0], data['udr2_gap_err'][0],
                  data['udr3_gap_err'][0]], capsize=8, width=WIDTH)

    axes[0].set_xticks(xvals)
    # axes[0].set_xticklabels(['Small','Medium','Large'])
    axes[0].set_xticklabels(['RL1','RL2','RL3'])
    axes[0].set_ylabel('RL reward - \nrule-based baseline')
    axes[0].set_title('Congestion control (CC)')
    # hide the right and top spines
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)         # ticks along the top edge are off
    # labels along the bottom edge are off


    small = [0.19]
    medium = [0.15]
    large = [0.06]

    small_err = [0.05]
    medium_err = [0.02]
    large_err = [0.03]
    axes[1].bar(xvals, [small[0], medium[0], large[0]],
            yerr=[small_err[0], medium_err[0], large_err[0]], capsize=8,
            width=WIDTH)

    axes[1].set_xticks(xvals)
    axes[1].set_xticklabels(['RL1','RL2','RL3'])
    axes[1].set_title('Adaptive bitrate (ABR)')
    # hide the right and top spines
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)         # ticks along the top edge are off
    # labels along the bottom edge are off


    small = [2.61]
    medium = [2.55]
    large = [2.21]

    small_err = [0.08]
    medium_err = [0.12]
    large_err = [0.14]
    axes[2].bar(xvals, [small[0], medium[0], large[0]],
            yerr=[small_err[0], medium_err[0], large_err[0]], capsize=8,
            width=WIDTH)

    axes[2].set_xticks(xvals)
    axes[2].set_xticklabels(['RL1','RL3','RL3'])
    axes[2].set_title('Load balancing (LB)')
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].set_ylim(2,)
    axes[2].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)         # ticks along the top edge are off
    # labels along the bottom edge are off
    fig.set_tight_layout(True)
    svg_file = os.path.join(ROOT, "motivation_udr_baseline_asymptotic_perf.svg")
    pdf_file = os.path.join(ROOT, "motivation_udr_baseline_asymptotic_perf.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))



    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    xvals = np.arange(0, 3) * (WIDTH + WIDTH / 3)
    data = pd.read_csv(os.path.join(ROOT, 'motivation_udr_baseline_percentage.csv'))
    assert isinstance(data, pd.DataFrame)
    axes[0].bar(
        xvals, [data['udr1_percent'][0] * 100, data['udr2_percent'][0] * 100,
                data['udr3_percent'][0] * 100],
        yerr=[data['udr1_percent_err'][0] * 100,
              data['udr2_percent_err'][0] * 100,
              data['udr3_percent_err'][0] * 100], capsize=8, width=WIDTH)

    axes[0].set_xticks(xvals)
    axes[0].set_xticklabels(['RL1','RL2','RL3'])
    axes[0].set_ylabel("% test traces where\nRL < non-ML baseline")
    axes[0].set_title('Congestion control (CC)')
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)         # ticks along the top edge are off
    # labels along the bottom edge are off


    small = [0.18]
    medium = [0.22]
    large = [0.32]

    small_err = [0.05]
    medium_err = [0.045]
    large_err = [0.063]
    axes[1].bar(xvals, [small[0] * 100, medium[0] * 100, large[0] * 100],
            yerr=[small_err[0] * 100, medium_err[0] * 100, large_err[0] * 100],
            capsize=8, width=WIDTH)
    axes[1].set_ylim(0, 45)
    axes[1].set_xticks(xvals)
    axes[1].set_xticklabels(['RL1','RL2','RL3'])
    axes[1].set_title('Adaptive bitrate (ABR)')
    # hide the right and top spines
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)         # ticks along the top edge are off
    # labels along the bottom edge are off


    small = [0.23]
    medium = [0.36]
    large = [0.38]

    small_err = [0.04]
    medium_err = [0.035]
    large_err = [0.058]
    axes[2].bar(xvals, [small[0] * 100, medium[0] * 100, large[0] * 100],
            yerr=[small_err[0] * 100, medium_err[0] * 100, large_err[0] * 100],
            capsize=8, width=WIDTH)

    axes[2].set_xticks(xvals)
    axes[2].set_xticklabels(['RL1','RL2','RL3'])
    axes[2].set_title('Load balancing (LB)')
    # hide the right and top spines
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)         # ticks along the top edge are off
    # labels along the bottom edge are off
    fig.set_tight_layout(True)

    svg_file = os.path.join(ROOT, "motivation_udr_baseline_percentage.svg")
    pdf_file = os.path.join(ROOT, "motivation_udr_baseline_percentage.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))


def motivation_original_aurora_baseline():
    plt.rcParams['font.size'] = 26
    plt.rcParams['axes.labelsize'] = 26
    plt.rcParams['axes.titlesize'] = 26
    plt.rcParams['legend.fontsize'] = 26

    width = 0.4
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    xvals = np.arange(0, 2) * (width + width / 2)
    data = pd.read_csv(os.path.join(ROOT, 'original_aurora_against_baseline_a.csv'))
    assert isinstance(data, pd.DataFrame)
    print(xvals)
    print(data['aurora_reward_syn'][0])
    print(data['aurora_reward_err_syn'][0])
    axes[0].bar(xvals[0], [data['aurora_reward_syn'][0] + 400],
            yerr=[data['aurora_reward_err_syn'][0]], capsize=8, width=width,
            bottom=-400)
    axes[0].bar(xvals[1], data['bbr_old_reward_syn'][0] + 400, color='C2',
            width=width, hatch=HATCHES[1], bottom=-400)

    axes[0].set_ylabel('Test reward on\nsynthetic envs')
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    # axes[0].set_xticks(xvals)
    # axes[0].set_xticklabels(["",""])
    axes[0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)# labels along the bottom edge are off

    axes[1].bar(xvals[0], [data['aurora_reward_cellular'][0]],
            yerr=[data['aurora_reward_err_cellular'][0]], capsize=8, width=width)
    axes[1].bar(xvals[1], data['bbr_old_reward_cellular'][0], color='C2',
            width=width, hatch=HATCHES[1])

    axes[1].set_ylabel('Test reward on\ntrace set Cellular')
    axes[1].set_xticks(xvals)
    axes[1].set_xticklabels(["",""])
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

    axes[2].bar(xvals[0], [data['aurora_reward_ethernet'][0]],
                yerr=[data['aurora_reward_err_ethernet'][0]], capsize=8,
                width=width, label='RL-based CC trained\nover synthetic envs')
    axes[2].bar(xvals[1], data['bbr_old_reward_ethernet'][0], color='C2',
                width=width, hatch=HATCHES[1], label='Rule-based\nbaseline BBR')

    axes[2].set_xticks(xvals)
    axes[2].set_xticklabels(["",""])
    axes[2].set_ylabel('Test reward on\ntrace set Ethernet')
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)# labels along the bottom edge are off

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels,  bbox_to_anchor=(0, 1.02, 1, 0.14), ncol=2, loc="upper center",
                borderaxespad=0, borderpad=0.3,  )
    fig.set_tight_layout(True)

    svg_file = os.path.join(ROOT, "motivation_original_aurora_baseline_a.svg")
    pdf_file = os.path.join(ROOT, "motivation_original_aurora_baseline_a.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))



    width = 0.4
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    xvals = np.arange(0, 3)  * 0.6 #(width + width / 3)
    data = pd.read_csv(os.path.join(ROOT, 'original_aurora_against_baseline_b.csv'))
    assert isinstance(data, pd.DataFrame)
    print(xvals)
    axes[0].bar(xvals[0], [data['train_cellular_test_cellular_reward'][0]],
            yerr=[data['train_cellular_test_cellular_reward_err'][0]], capsize=8, width=width)
    axes[0].bar(xvals[1], [data['train_ethernet_test_cellular_reward'][0]],
            yerr=[data['train_ethernet_test_cellular_reward_err'][0]], capsize=8, width=width, hatch=HATCHES[0])
    axes[0].bar(xvals[2], data['bbr_old_test_cellular_reward'][0], width=width, hatch=HATCHES[1])
    axes[0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)# labels along the bottom edge are off
    # print(data['train_cellular_test_cellular_reward'][0], data['train_ethernet_test_cellular_reward'][0],  data['bbr_old_test_cellular_reward'][0])

    # axes[0].set_ylabel('Test reward on\ntrace set Cellular')
    axes[0].set_ylabel('Cellular test reward')
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].set_xticks(xvals)
    axes[0].set_xticklabels(["","", ""])
    axes[0].set_ylim(50, )

    axes[1].bar(xvals[0], [data['train_cellular_test_ethernet_reward'][0]],
            yerr=[data['train_cellular_test_ethernet_reward_err'][0]],
            capsize=8, width=width, label='Cellular-trained CC')
            # label='RL-based CC trained\nover trace set Cellular')
    axes[1].bar(xvals[1], [data['train_ethernet_test_ethernet_reward'][0]],
            yerr=[data['train_ethernet_test_ethernet_reward_err'][0]],
            capsize=8, width=width, hatch=HATCHES[0],
            label='Ethernet-trained CC')
            # label='RL-based CC trained\nover trace set Ethernet')
    axes[1].bar(xvals[2], data['bbr_old_test_ethernet_reward'][0], width=width,
                hatch=HATCHES[1], label='Baseline BBR')
 # label='Rule-based\nbaseline BBR')
    print(data['train_cellular_test_ethernet_reward'][0], data['train_ethernet_test_ethernet_reward'][0],  data['bbr_old_test_ethernet_reward'][0])

    # axes[1].set_ylabel('Test reward on\ntrace set Ethernet')
    axes[1].set_ylabel('Ethernet test reward')
    axes[1].set_xticks(xvals)
    axes[1].set_xticklabels(["","", ""])
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].set_ylim(200, )
    axes[1].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)# labels along the bottom edge are off

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels,  bbox_to_anchor=(0, 1.02, 1, 0.08), ncol=3, loc="upper center",
                borderaxespad=0, borderpad=0.3, columnspacing=0.8, handletextpad=0.6, handlelength=1.5)
    fig.set_tight_layout(True)

    svg_file = os.path.join(ROOT, "motivation_original_aurora_baseline_b.svg")
    pdf_file = os.path.join(ROOT, "motivation_original_aurora_baseline_b.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))


def gap_vs_improvement_scatter():
    plt.rcParams['font.size'] = 26
    plt.rcParams['axes.labelsize'] = 26
    plt.rcParams['axes.titlesize'] = 26
    plt.rcParams['legend.fontsize'] = 26
    fig, axes = plt.subplots(1, 2, figsize=(13, 3))
    data_4_cl_3 = [10.41, 14.2, 10.7, 10.73, 8.22, 10.5, 13.7, 14.1, 12.3,
                   10.1, 9.7, 8.45, 9.52, 9.03, 11.8, 8.45, 11.55, 12.85,
                   13.69, 11.58, 7.11, 11.24, 11.74, 17.4,
                   8.4, 8.83, 8.5, 8.05, 5.42, 11.39, 8.15, 8.83, 14.0,
                   17.23, 12.5, 12.0, 8.75, 17.0, 13.83, 10.71]
    data_4_y = [1.51, 2.9, 2.62, 5.13, 4.18, 1.06, 4.74, 4.03, 1.28, 2.26,
                2.49, 4.27, 1.46, 0.97, 4.23, 3.07, 0.73, 3.65, 5.42, 7.03,
                1.09, 3.39, 4.21, 5.12, 0.28, 3.52, 3.63, 3.16, 1.24, 3.13,
                1.22, 1.15, 4.3, 2.02, 3.52, 6.03, 2.3, 7.72, 3.43, 5.12]
    # coeff, _ = kendalltau(data_4_cl_3, data_4_y)
    coeff, _ = pearsonr(data_4_cl_3, data_4_y)
    axes[0].scatter(data_4_cl_3, data_4_y)
    axes[0].set_ylabel('Training\nimprovement')
    axes[0].set_xlabel('Current model\'s performance\n(Strawman 3)')
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    # axes[0].set_title('Kendall\'s rank coefficient: {:.2f}'.format(coeff))
    axes[0].set_title('Pearson correlation coef.: {:.2f}'.format(coeff))

    data_4_x = [2.31, 3.7, 3.5, 4.63, 3.52, 2.1, 5.1, 4.9, 3.3, 3.9,
                2.4, 3.2, 2.62, 1.23, 4.31, 1.25, 1.05, 4.65, 5.55, 6.28,
                1.61, 4.54, 5.34, 6.24, 0.6, 2.63, 3.2, 2.15, 1.22, 2.8,
                0.65, 1.73, 3.7, 5.13, 3.2, 7.8, 1.25, 8.6, 4.23, 4.01]

    data_4_y = [1.51, 2.9, 2.62, 5.13, 4.18, 1.06, 4.74, 4.03, 1.28, 2.26,
                2.49, 4.27, 1.46, 0.97, 4.23, 3.07, 0.73, 3.65, 5.42, 7.03,
                1.09, 3.39, 4.21, 5.12, 0.28, 3.52, 3.63, 3.16, 1.24, 3.13,
                1.22, 1.15, 4.3, 2.02, 3.52, 6.03, 2.3, 7.72, 3.43, 5.12]
    # coeff, _ = kendalltau(data_4_x, data_4_y)
    coeff, _ = pearsonr(data_4_x, data_4_y)
    axes[1].scatter(data_4_x, data_4_y, c='C1')
    # axes[1].set_ylabel('Training\nimprovement')
    axes[1].set_xlabel('Gap-to-baseline of current model\n(Genet)')
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    # axes[1].set_title('Kendall\'s rank coefficient: {:.2f}'.format(coeff))
    axes[1].set_title('Pearson correlation coef.: {:.2f}'.format(coeff))
    # fig.set_tight_layout(True)
    svg_file = os.path.join(ROOT, "design_gap_vs_improvement_abr.svg")
    pdf_file = os.path.join(ROOT, "design_gap_vs_improvement_abr.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))

    fig, axes = plt.subplots(1, 2, figsize=(13, 3))
    data = pd.read_csv('../../figs_sigcomm22/cl3_gap_improvement.csv')
    assert isinstance(data, pd.DataFrame)
    # coeff, _ = kendalltau(data['cl3_metric'], data['improvement'])
    coeff, _ = pearsonr(data['cl3_metric'], data['improvement'])
    axes[0].scatter(data['cl3_metric'], data['improvement'])
    axes[0].set_ylabel('Training\nimprovement')
    axes[0].set_xlabel('Current model\'s performance\n(Strawman 3)')
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    # axes[0].set_title('Kendall\'s rank coefficient: {:.2f}'.format(coeff))
    axes[0].set_title('Pearson correlation coef.: {:.2f}'.format(coeff))

    data = pd.read_csv('../../figs_sigcomm22/genet_gap_improvement.csv')
    assert isinstance(data, pd.DataFrame)
    # coeff, _ = kendalltau(data['genet_metric'], data['improvement'])
    coeff, _ = pearsonr(data['genet_metric'], data['improvement'])
    axes[1].scatter(data['genet_metric'], data['improvement'], c='C1')
    # axes[1].set_ylabel('Training\nimprovement')
    axes[1].set_xlabel('Gap-to-baseline of current model\n(Genet)')
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    # axes[1].set_title('Kendall\'s rank coefficient: {:.2f}'.format(coeff))
    axes[1].set_title('Pearson correlation coef.: {:.2f}'.format(coeff))
    # fig.set_tight_layout(True)
    svg_file = os.path.join(ROOT, "design_gap_vs_improvement_cc.svg")
    pdf_file = os.path.join(ROOT, "design_gap_vs_improvement_cc.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))


def default_synthetic_envs():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 11))
    plt.rcParams['font.size'] = 23
    plt.rcParams['axes.labelsize'] = 23
    plt.rcParams['axes.titlesize'] = 23
    plt.rcParams['legend.fontsize'] = 23
    # 'vary_delay',
    for ax, dim in zip(axes.flatten(), ['vary_loss', 'vary_delay', 'vary_T_s', 'vary_queue']):
        data = pd.read_csv('../../figs_sigcomm22/default_syn_envs_{}.csv'.format(dim))
        assert isinstance(data, pd.DataFrame)
        ax.plot(data['vals'], data['bbr_old_rewards'], ls='--', label='BBR')
        ax.fill_between(data['vals'], data['bbr_old_rewards'] - data['bbr_old_reward_errs'],
                       data['bbr_old_rewards'] + data['bbr_old_reward_errs'], color='C0', alpha=0.1)
        ax.plot(data['vals'], data['cubic_rewards'], ls=':', label='Cubic')
        ax.fill_between(data['vals'], data['cubic_rewards'] - data['cubic_reward_errs'],
                       data['cubic_rewards'] + data['cubic_reward_errs'], color='C1',alpha=0.1)
        ax.plot(data['vals'], data['genet_bbr_old_rewards'], label='GENET')
        ax.fill_between(data['vals'], data['genet_bbr_old_rewards'] - data['genet_bbr_old_reward_errs'],
                       data['genet_bbr_old_rewards'] + data['genet_bbr_old_reward_errs'], color='C2', alpha=0.1)
        ax.set_ylabel('Test reward')
        if dim == 'vary_loss':
            ax.set_xlabel('Random packet loss rate (%)')
        elif dim == 'vary_bw':
            ax.set_xlabel('Max bandwidth (Mbps)')
        elif dim == 'vary_T_s':
            ax.set_xlabel('Bandwidth change interval (s)')
        elif dim == 'vary_queue':
            ax.set_xlabel('Queue (BDP)')
        else:
            ax.set_xlabel('Link RTT (ms)')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,  bbox_to_anchor=(0, 1.02, 1, 0.), ncol=3, loc="upper center",
                borderaxespad=0, borderpad=0.3)
    fig.set_tight_layout(True)
    svg_file = os.path.join(ROOT, "evaluation_default_synthetic_envs_cc.svg")
    pdf_file = os.path.join(ROOT, "evaluation_default_synthetic_envs_cc.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))


def bo_efficiency():
    plt.rcParams['font.size'] = 30
    plt.rcParams['axes.labelsize'] = 30
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['legend.fontsize'] = 30
    fig, ax = plt.subplots(figsize=(8, 6))

    data = pd.read_csv('../../figs_sigcomm22/bo_efficiency.csv')
    assert isinstance(data, pd.DataFrame)
    ax.plot(data['rand_num_samples'], data['rand_max'], ls='--', lw=5,
            label='Random\nexploration')
    ax.fill_between(data['rand_num_samples'], data['rand_max'] - data['rand_max_err'],
                   data['rand_max'] + data['rand_max_err'], color='C0', alpha=0.2)
    ax.plot(data['bo_num_samples'], data['bo_max'], lw=5, color='C2',
            label='BO-based\nexploration')
    ax.fill_between(data['bo_num_samples'], data['bo_max'] - data['bo_max_err'],
                    data['bo_max'] + data['bo_max_err'], color='C2', alpha=0.2)
    ax.annotate("BO-based\nexploration", (16, 155))
    ax.annotate("Random\nexploration", (55, 58))
    ax.set_ylabel('Gap to baseline of\nthe chosen config')
    ax.set_xlabel('# of samples explored')
    ax.set_ylim(-100, )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # ax.legend(loc='lower right')
    fig.set_tight_layout(True)

    svg_file = os.path.join(ROOT, "evaluation_bo_efficiency.svg")
    pdf_file = os.path.join(ROOT, "evaluation_bo_efficiency.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))


    x_random = [1, 2, 3, 4, 5, 10, 15, 20, 40, 50, 100]
    y_random  = [0.5, 1.3, 1.39, 1.65, 2.1, 2.5, 3.75, 3.82, 4.01, 4.04, 4.15]

    random_low_bnd = [0.2405230117831404, 1.1075354766663823,
            0.9375536678173122, 1.193191763046101, 1.8222653231879538,
            2.2233251675090537, 3.4240923240672556, 3.620389860409379,
            3.7148179001361843, 3.8574181354916397, 3.93297561264385]
    random_up_bnd = [0.7594769882168596, 1.4924645233336178,
            1.8424463321826876, 2.1068082369538987, 2.3777346768120466,
            2.7766748324909463, 4.075907675932744, 4.01961013959062,
            4.305182099863815, 4.222581864508361, 4.3670243873561505]


    x_bo = [1, 2, 3, 4, 5, 10, 15]
    y_bo = [0.4, 1.9, 2.4, 2.9, 3.6, 4.17, 4.29]

    bo_low_bnd = [0.15967081394587998, 1.6544254047719775, 2.0099111556708746,
            2.5843046656492596, 3.0411887398527497, 3.7767270231358676,
            3.788957274453827]
    bo_up_bnd =[0.64032918605412, 2.1455745952280223, 2.790088844329125,
            3.21569533435074, 4.15881126014725, 4.563272976864132,
            4.791042725546173]

    fig, ax = plt.subplots(figsize=(8, 6))

    assert isinstance(data, pd.DataFrame)
    ax.plot(x_random, y_random, ls='--', lw=5, label='Random\nexploration')
    ax.fill_between(x_random, random_low_bnd, random_up_bnd, color='C0', alpha=0.2)
    ax.plot(x_bo, y_bo, lw=5, color='C2', label='BO-based\nexploration')
    ax.fill_between(x_bo, bo_low_bnd, bo_up_bnd, color='C2', alpha=0.2)
    ax.annotate("BO-based\nexploration", (16, 4.5))
    ax.annotate("Random\nexploration", (50, 2.8))
    ax.set_ylabel('Gap to baseline of\nthe chosen config')
    ax.set_xlabel('# of samples explored')
    # ax.set_ylim(-100, )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # ax.legend(loc='lower right')
    fig.set_tight_layout(True)

    svg_file = os.path.join(ROOT, "evaluation_bo_efficiency_abr.svg")
    pdf_file = os.path.join(ROOT, "evaluation_bo_efficiency_abr.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))


def percentages():
    genet_percent_better_than_bbr = (121 * 0.835 + 112 * (0.598 + 0.714)/2) / (121 + 112)
    genet_percent_better_than_cubic = (121 * 0.57 + 112 * (0.393 + 0.518) / 2) / (121 + 112)

    udr1_percent_better_than_bbr = (121 * 0.802 + 112 * (0.438 + 0.616) / 2) / (121 + 112)
    udr1_percent_better_than_cubic = (121 * 0.504 + 112 * (0.304 + 0.473) / 2) / (121 + 112)

    udr2_percent_better_than_bbr = (121 * 0.81 + 112 * (0.464 + 0.67)/2) / (121 + 112)
    udr2_percent_better_than_cubic = (121 * 0.438 + 112 * (0.286 + 0.518)/2) / (121 + 112)

    udr3_percent_better_than_bbr = (121 * 0.777 + 112 * (0.339 + 0.661)/2) / (121 + 112)
    udr3_percent_better_than_cubic = (121 * 0.421 + 112 * (0.241 + 0.438)/2) / (121 + 112)

    print('genet', round(genet_percent_better_than_bbr, 2), round(genet_percent_better_than_cubic, 2))
    print('udr1', round(udr1_percent_better_than_bbr, 2), round(udr1_percent_better_than_cubic, 2))
    print('udr2', round(udr2_percent_better_than_bbr, 2), round(udr2_percent_better_than_cubic, 2))
    print('udr3', round(udr3_percent_better_than_bbr, 2), round(udr3_percent_better_than_cubic, 2))

def baseline_choice():
    mpc = 0.67
    mpc_err = 0.02
    genet_mpc = 0.86
    genet_mpc_err = 0.01
    bba = 0.57
    bba_err = 0.008
    genet_bba = 0.83
    genet_bba_err = 0.02

    plt.rcParams['font.size'] = 36
    plt.rcParams['axes.labelsize'] = 36
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['legend.fontsize'] = 36
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([1, 3.5], [mpc, bba]) #, yerr=[mpc_err, bba_err], capsize=8)

    ax.bar([2.2, 4.7], [genet_mpc, genet_bba], yerr=[genet_mpc_err, genet_bba_err],
                  color='C2', capsize=8)
    ax.set_ylabel('Test reward')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False         # ticks along the top edge are off
    )# labels along the bottom edge are off
    ax.set_xticks([1, 2.2, 3.5, 4.7])
    ax.set_xticklabels(['MPC','Genet\n(MPC)', 'BBA', 'Genet\n(BBA)'])

    # assert isinstance(data, pd.DataFrame)

    # ax.legend(loc='lower right')
    # fig.set_tight_layout(True)

    svg_file = os.path.join(ROOT, "evaluation_baseline_choice_abr.svg")
    pdf_file = os.path.join(ROOT, "evaluation_baseline_choice_abr.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))


    bbr = 192
    genet_bbr = 211
    cubic = 97.16 #  33.99   802.69  0.02
    genet_cubic = 195

    bbr_err = 0
    genet_bbr_err = 7.10
    cubic_err = 0
    genet_cubic_err = 9.17

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([1, 3.5], [bbr, cubic]) #, yerr=[bbr_err, cubic_err], capsize=8)

    ax.bar([2.2, 4.7], [genet_bbr, genet_cubic], yerr=[genet_bbr_err, genet_cubic_err],
                  color='C2', capsize=8)
    ax.set_ylabel('Test reward')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)         # ticks along the top edge are off
    ax.set_xticks([1, 2.2, 3.5, 4.7])
    ax.set_xticklabels(['BBR','Genet\n(BBR)', 'Cubic', 'Genet\n(Cubic)'])

    svg_file = os.path.join(ROOT, "evaluation_baseline_choice_cc.svg")
    pdf_file = os.path.join(ROOT, "evaluation_baseline_choice_cc.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))




    llf = 490.88
    genet_llf = 590.27
    c3 = 506.65
    genet_c3 = 604.23

    llf_err = 35.15
    genet_llf_err = 57.17
    c3_err = 43.23
    genet_c3_err = 46.21

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar([1, 3.5], [llf, c3]) #, yerr=[llf_err, c3_err], capsize=8)
    ax.bar([2.2, 4.7], [genet_llf, genet_c3], yerr=[genet_llf_err, genet_c3_err],
                  color='C2', capsize=8)
    ax.set_ylabel('Test reward')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)         # ticks along the top edge are off
    ax.set_xticks([1, 2.2, 3.5, 4.7])
    ax.set_xticklabels(['LLF','Genet\n(LLF)', 'C3', 'Genet\n(C3)'])

    svg_file = os.path.join(ROOT, "evaluation_baseline_choice_lb.svg")
    pdf_file = os.path.join(ROOT, "evaluation_baseline_choice_lb.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))


def better_fraction():
    plt.rcParams['font.size'] = 35
    plt.rcParams['axes.labelsize'] = 35
    plt.rcParams['axes.titlesize'] = 35
    plt.rcParams['legend.fontsize'] = 35
    genet_cubic = 0.2
    genet_bbr = 0.19
    rl1_cubic = 0.49
    rl1_bbr = 0.35
    rl2_cubic = 0.41
    rl2_bbr = 0.39
    rl3_cubic = 0.47
    rl3_bbr = 0.43

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([1, 2, 3, 4], np.array([1-rl1_cubic, 1-rl2_cubic, 1-rl3_cubic, 1 - genet_cubic]) * 100, marker='s', lw=4, ms=8)
    ax.plot([1, 2, 3, 4], np.array([1- rl1_bbr, 1- rl2_bbr, 1-rl3_bbr, 1 - genet_bbr]) * 100, marker='o', lw=4, ms=8)
    ax.annotate("Baseline: Cubic", (1.5, 0.44*100))
    ax.annotate("Baseline: BBR", (1.5, 0.67*100))
    ax.set_ylabel('% traces better than baselines\n(represented by a line)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)         # ticks along the top edge are off
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['RL1', 'RL2', 'RL3', 'Genet'])
    ax.set_ylim(0,)

    svg_file = os.path.join(ROOT, "evaluation_better_trace_percent.svg")
    pdf_file = os.path.join(ROOT, "evaluation_better_trace_percent.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))


    genet_cubic = 0.13
    genet_bbr = 0.07
    rl1_cubic = 0.25
    rl1_bbr = 0.39
    rl2_cubic = 0.46
    rl2_bbr = 0.29
    rl3_cubic = 0.46
    rl3_bbr = 0.34

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot([1, 2, 3, 4], np.array([1-rl1_cubic, 1-rl2_cubic, 1-rl3_cubic, 1 - genet_cubic]) * 100, marker='s', lw=4, ms=8)
    ax.plot([1, 2, 3, 4], np.array([1- rl1_bbr, 1- rl2_bbr, 1-rl3_bbr, 1 - genet_bbr]) * 100, marker='o', lw=4, ms=8)
    ax.annotate("Baseline: MPC", (1.5, 0.41*100))
    ax.annotate("Baseline: BBA", (1.5, 0.75*100))
    ax.set_ylabel('% traces better than baselines\n(represented by a line)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)         # ticks along the top edge are off
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['RL1', 'RL2', 'RL3', 'Genet'])
    ax.set_ylim(0,)

    svg_file = os.path.join(ROOT, "evaluation_better_trace_percent_abr.svg")
    pdf_file = os.path.join(ROOT, "evaluation_better_trace_percent_abr.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))

if __name__ == '__main__':
    # motivation_udr_baseline()
    # motivation_original_aurora_baseline()
    # gap_vs_improvement_scatter()
    # default_synthetic_envs()
    # bo_efficiency()
    # percentages()
    # baseline_choice()
    better_fraction()
