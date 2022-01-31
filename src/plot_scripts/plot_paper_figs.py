import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau

ROOT = '../../figs_sigcomm22'

plt.style.use('seaborn-deep')
plt.rcParams['font.family'] = 'Arial' #'Times New Roman' or
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['svg.fonttype'] = 'none'

HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

WIDTH = 0.3

def motivation_udr_baseline():
    # motivation figures
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    xvals = np.arange(0, 3) * (WIDTH + WIDTH / 3)
    data = pd.read_csv(os.path.join(ROOT, 'motivation_udr_baseline_asymptotic_perf.csv'))
    bars = axes[0].bar(xvals,
           [data['udr1_gap'][0], data['udr2_gap'][0], data['udr3_gap'][0]],
            yerr=[data['udr1_gap_err'][0], data['udr2_gap_err'][0],
                  data['udr3_gap_err'][0]], capsize=8, width=WIDTH)
    for bar, pat in zip(bars, HATCHES):
        bar.set_hatch(pat)

    axes[0].set_xticks(xvals)
    axes[0].set_xticklabels(['Small','Medium','Large'])
    axes[0].set_ylabel('RL reward - \nrule-based baseline')
    axes[0].set_title('Congestion control (CC)')
    # hide the right and top spines
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)


    small = [0.19]
    medium = [0.15]
    large = [0.06]

    small_err = [0.05]
    medium_err = [0.02]
    large_err = [0.03]
    bars = axes[1].bar(xvals, [small[0], medium[0], large[0]],
            yerr=[small_err[0], medium_err[0], large_err[0]], capsize=8,
            width=WIDTH)
    for bar, pat in zip(bars, HATCHES):
        bar.set_hatch(pat)

    axes[1].set_xticks(xvals)
    axes[1].set_xticklabels(['Small','Medium','Large'])
    axes[1].set_title('Adaptive bitrate (ABR)')
    # hide the right and top spines
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)



    small = [2.61]
    medium = [2.55]
    large = [2.21]

    small_err = [0.08]
    medium_err = [0.12]
    large_err = [0.14]
    bars = axes[2].bar(xvals, [small[0], medium[0], large[0]],
            yerr=[small_err[0], medium_err[0], large_err[0]], capsize=8,
            width=WIDTH)
    for bar, pat in zip(bars, HATCHES):
        bar.set_hatch(pat)

    axes[2].set_xticks(xvals)
    axes[2].set_xticklabels(['Small','Medium','Large'])
    axes[2].set_title('Load balancing (LB)')
    # hide the right and top spines
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    svg_file = os.path.join(ROOT, "motivation_udr_baseline_asymptotic_perf.svg")
    pdf_file = os.path.join(ROOT, "motivation_udr_baseline_asymptotic_perf.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))



    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    xvals = np.arange(0, 3) * (WIDTH + WIDTH / 3)
    data = pd.read_csv(os.path.join(ROOT, 'motivation_udr_baseline_percentage.csv'))
    bars = axes[0].bar(xvals,
           [data['udr1_percent'][0], data['udr2_percent'][0], data['udr3_percent'][0]],
            yerr=[data['udr1_percent_err'][0], data['udr2_percent_err'][0],
                  data['udr3_percent_err'][0]], capsize=8, width=WIDTH)

    for bar, pat in zip(bars, HATCHES):
        bar.set_hatch(pat)
    print(xvals)

    axes[0].set_xticks(xvals)
    axes[0].set_xticklabels(['Small','Medium','Large'])
    axes[0].set_ylabel("% test traces where\nRL < non-ML baseline")
    axes[0].set_title('Congestion control (CC)')
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)


    small = [0.18]
    medium = [0.22]
    large = [0.32]

    small_err = [0.05]
    medium_err = [0.045]
    large_err = [0.063]
    bars = axes[1].bar(xvals, [small[0], medium[0], large[0]],
            yerr=[small_err[0], medium_err[0], large_err[0]], capsize=8,
            width=WIDTH)
    for bar, pat in zip(bars, HATCHES):
        bar.set_hatch(pat)

    axes[1].set_xticks(xvals)
    axes[1].set_xticklabels(['Small','Medium','Large'])
    axes[1].set_title('Adaptive bitrate (ABR)')
    # hide the right and top spines
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)



    small = [0.23]
    medium = [0.36]
    large = [0.38]

    small_err = [0.04]
    medium_err = [0.035]
    large_err = [0.058]
    bars = axes[2].bar(xvals, [small[0], medium[0], large[0]],
            yerr=[small_err[0], medium_err[0], large_err[0]], capsize=8,
            width=WIDTH)
    for bar, pat in zip(bars, HATCHES):
        bar.set_hatch(pat)

    axes[2].set_xticks(xvals)
    axes[2].set_xticklabels(['Small','Medium','Large'])
    axes[2].set_title('Load balancing (LB)')
    # hide the right and top spines
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)


    svg_file = os.path.join(ROOT, "motivation_udr_baseline_percentage.svg")
    pdf_file = os.path.join(ROOT, "motivation_udr_baseline_percentage.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))


def motivation_original_aurora_baseline():
    width = 0.4
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    xvals = np.arange(0, 2) * (width + width / 2)
    data = pd.read_csv(os.path.join(ROOT, 'original_aurora_against_baseline_a.csv'))
    print(xvals)
    print(data['aurora_reward_syn'][0])
    print(data['aurora_reward_err_syn'][0])
    axes[0].bar(xvals[0], [data['aurora_reward_syn'][0]],
            yerr=[data['aurora_reward_err_syn'][0]], capsize=8, width=width, hatch=HATCHES[0])
    axes[0].bar(xvals[1], data['bbr_old_reward_syn'][0], width=width, hatch=HATCHES[1])

    axes[0].set_ylabel('Test reward on synthetic envs')
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].set_xticks(xvals)
    axes[0].set_xticklabels(["",""])

    axes[1].bar(xvals[0], [data['aurora_reward_cellular'][0]],
            yerr=[data['aurora_reward_err_cellular'][0]], capsize=8, width=width, hatch=HATCHES[0])
    axes[1].bar(xvals[1], data['bbr_old_reward_cellular'][0], width=width, hatch=HATCHES[1])

    axes[1].set_ylabel('Test reward on trace set Cellular')
    axes[1].set_xticks(xvals)
    axes[1].set_xticklabels(["",""])
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)

    axes[2].bar(xvals[0], [data['aurora_reward_ethernet'][0]],
            yerr=[data['aurora_reward_err_ethernet'][0]], capsize=8, width=width, hatch=HATCHES[0], label='Aurora')
    axes[2].bar(xvals[1], data['bbr_old_reward_ethernet'][0], width=width, hatch=HATCHES[1], label='BBR')

    axes[2].set_xticks(xvals)
    axes[2].set_xticklabels(["",""])
    axes[2].set_ylabel('Test reward on trace set Ethernet')
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels,  bbox_to_anchor=(0, 1.02, 1, 0.0), ncol=2, loc="upper center",
                borderaxespad=0)
    # fig.set_tight_layout(True)

    svg_file = os.path.join(ROOT, "motivation_original_aurora_baseline_a.svg")
    pdf_file = os.path.join(ROOT, "motivation_original_aurora_baseline_a.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))



    width = 0.4
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    xvals = np.arange(0, 3)  * 0.6 #(width + width / 3)
    data = pd.read_csv(os.path.join(ROOT, 'original_aurora_against_baseline_b.csv'))
    print(xvals)
    axes[0].bar(xvals[0], [data['train_cellular_test_cellular_reward'][0]],
            yerr=[data['train_cellular_test_cellular_reward_err'][0]], capsize=8, width=width, hatch=HATCHES[0])
    axes[0].bar(xvals[1], [data['train_ethernet_test_cellular_reward'][0]],
            yerr=[data['train_ethernet_test_cellular_reward_err'][0]], capsize=8, width=width, hatch=HATCHES[1])
    axes[0].bar(xvals[2], data['bbr_old_test_cellular_reward'][0], width=width, hatch=HATCHES[2])

    axes[0].set_ylabel('Test reward on\ntrace set Cellular')
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].set_xticks(xvals)
    axes[0].set_xticklabels(["","", ""])
    axes[0].set_ylim(50, )

    axes[1].bar(xvals[0], [data['train_cellular_test_ethernet_reward'][0]],
            yerr=[data['train_cellular_test_ethernet_reward_err'][0]],
            capsize=8, width=width, hatch=HATCHES[0],
            label='RL-based CC trained\nover trace set Cellular')
    axes[1].bar(xvals[1], [data['train_ethernet_test_ethernet_reward'][0]],
            yerr=[data['train_ethernet_test_ethernet_reward_err'][0]],
            capsize=8, width=width, hatch=HATCHES[1],
            label='RL-based CC trained\nover trace set Ethernet')
    axes[1].bar(xvals[2], data['bbr_old_test_ethernet_reward'][0], width=width,
                hatch=HATCHES[2], label='Rule-based\nbaseline BBR')

    axes[1].set_ylabel('Test reward on\ntrace set Ethernet')
    axes[1].set_xticks(xvals)
    axes[1].set_xticklabels(["","", ""])
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].set_ylim(200, )


    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels,  bbox_to_anchor=(0, 1.02, 1, 0.07), ncol=3, loc="upper center",
                borderaxespad=0, borderpad=0.1)
    fig.set_tight_layout(True)

    svg_file = os.path.join(ROOT, "motivation_original_aurora_baseline_b.svg")
    pdf_file = os.path.join(ROOT, "motivation_original_aurora_baseline_b.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))


def gap_vs_improvement_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    data_4_cl_3 = [10.41, 14.2, 10.7, 10.73, 8.22, 10.5, 13.7, 14.1, 12.3,
                   10.1, 9.7, 8.45, 9.52, 9.03, 11.8, 8.45, 11.55, 12.85,
                   13.69, 11.58, 7.11, 11.24, 11.74, 17.4,
                   8.4, 8.83, 8.5, 8.05, 5.42, 11.39, 8.15, 8.83, 14.0,
                   17.23, 12.5, 12.0, 8.75, 17.0, 13.83, 10.71]
    data_4_y = [1.51, 2.9, 2.62, 5.13, 4.18, 1.06, 4.74, 4.03, 1.28, 2.26,
                2.49, 4.27, 1.46, 0.97, 4.23, 3.07, 0.73, 3.65, 5.42, 7.03,
                1.09, 3.39, 4.21, 5.12, 0.28, 3.52, 3.63, 3.16, 1.24, 3.13,
                1.22, 1.15, 4.3, 2.02, 3.52, 6.03, 2.3, 7.72, 3.43, 5.12]
    coeff, _ = kendalltau(data_4_cl_3, data_4_y)
    axes[0].scatter(data_4_cl_3, data_4_y)
    axes[0].set_ylabel('Training improvement')
    axes[0].set_xlabel('Current model\'s performance\n(Strawman 3)')
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].set_title('Kendall\'s rank coefficnet: {:.2f}'.format(coeff))

    data_4_x = [2.31, 3.7, 3.5, 4.63, 3.52, 2.1, 5.1, 4.9, 3.3, 3.9,
                2.4, 3.2, 2.62, 1.23, 4.31, 1.25, 1.05, 4.65, 5.55, 6.28,
                1.61, 4.54, 5.34, 6.24, 0.6, 2.63, 3.2, 2.15, 1.22, 2.8,
                0.65, 1.73, 3.7, 5.13, 3.2, 7.8, 1.25, 8.6, 4.23, 4.01]

    data_4_y = [1.51, 2.9, 2.62, 5.13, 4.18, 1.06, 4.74, 4.03, 1.28, 2.26,
                2.49, 4.27, 1.46, 0.97, 4.23, 3.07, 0.73, 3.65, 5.42, 7.03,
                1.09, 3.39, 4.21, 5.12, 0.28, 3.52, 3.63, 3.16, 1.24, 3.13,
                1.22, 1.15, 4.3, 2.02, 3.52, 6.03, 2.3, 7.72, 3.43, 5.12]
    coeff, _ = kendalltau(data_4_x, data_4_y)
    axes[1].scatter(data_4_x, data_4_y)
    axes[1].set_ylabel('Training improvement')
    axes[1].set_xlabel('Gap-to-baseline of current model\n(Genet)')
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].set_title('Kendall\'s rank coefficnet: {:.2f}'.format(coeff))
    svg_file = os.path.join(ROOT, "design_gap_vs_improvement_abr.svg")
    pdf_file = os.path.join(ROOT, "design_gap_vs_improvement_abr.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    data = pd.read_csv('../../figs_sigcomm22/cl3_gap_improvement.csv')
    coeff, _ = kendalltau(data['cl3_metric'], data['improvement'])
    axes[0].scatter(data['cl3_metric'], data['improvement'])
    axes[0].set_ylabel('Training improvement')
    axes[0].set_xlabel('Current model\'s performance\n(Strawman 3)')
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].set_title('Kendall\'s rank coefficnet: {:.2f}'.format(coeff))

    data = pd.read_csv('../../figs_sigcomm22/genet_gap_improvement.csv')
    coeff, _ = kendalltau(data['genet_metric'], data['improvement'])
    axes[1].scatter(data['genet_metric'], data['improvement'])
    axes[1].set_ylabel('Training improvement')
    axes[1].set_xlabel('Current model\'s performance\n(Strawman 3)')
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].set_title('Kendall\'s rank coefficnet: {:.2f}'.format(coeff))
    svg_file = os.path.join(ROOT, "design_gap_vs_improvement_cc.svg")
    pdf_file = os.path.join(ROOT, "design_gap_vs_improvement_cc.pdf")
    fig.savefig(svg_file, bbox_inches='tight')
    os.system("inkscape {} --export-pdf={}".format(svg_file, pdf_file))
    os.system("pdfcrop --margins 1 {} {}".format(pdf_file, pdf_file))


if __name__ == '__main__':
    # motivation_original_aurora_baseline()
    gap_vs_improvement_scatter()
