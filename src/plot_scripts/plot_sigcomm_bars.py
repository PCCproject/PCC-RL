import os

import matplotlib.pyplot as plt
import numpy as np
from common.utils import compute_std_of_mean

SAVE_ROOT = '../../figs_sigcomm22'


# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 42
plt.rcParams['axes.labelsize'] = 42
plt.rcParams['legend.fontsize'] = 42
# plt.rcParams['legend.columnspacing'] = 0.5
# plt.rcParams['legend.labelspacing'] = 0.002
plt.rcParams['figure.figsize'] = (11, 9)


cl1_cellular_rewards = [-185.07, -161.79, -195.02]
cl1_ethernet_rewards = [113.85, 88.62, 125.57]
cl2_cellular_rewards = [-161.34, -194.27, -161.34]
cl2_ethernet_rewards = [179.74, 167.56, 186.60]
# genet_ethernet_rewards = [220, 200]
genet_ethernet_rewards = [238, 212]
genet_cellular_rewards = [-205, -236, -210]
genet_cellular_rewards = [378, 347, 372]
real_default_ethernet_rewards = [171, 179, 167]
# real_default_cellular_rewards = [-229, -263, -225]
real_default_cellular_rewards = [355, 361, 365]
udr1_ethernet_rewards = [128, 139, 83]
udr2_ethernet_rewards = [144, 142]
udr3_ethernet_rewards = [177, 173, 92]
udr3_real_5percent_ethernet_rewards = [177.2, 209.8, 95.2]
udr3_real_10percent_ethernet_rewards = [139, 175, 173]
udr3_real_20percent_ethernet_rewards = [133, 125, 151]
udr3_real_50percent_ethernet_rewards = [162, 124, 78]
udr1_cellular_rewards = [369, 293, 212]
udr2_cellular_rewards = [349, 321, 317]
udr3_cellular_rewards = [335, 359, 349]

# bbr_ethernet_rewards = [188]
bbr_ethernet_rewards = [209]
bbr_cellular_rewards = [238]

copa_ethernet_rewards = [211]
copa_cellular_rewards = [238]

cubic_cellular_rewards = [342]
cubic_ethernet_rewards = [107]

genet_real_1percent_ethernet_rewards = [214, 191]
genet_real_4percent_ethernet_rewards = [213, 219, 216]
genet_real_5percent_ethernet_rewards = [219.3, 213.4, 226.5]
genet_real_6percent_ethernet_rewards = [223, 208]
genet_real_20percent_ethernet_rewards = [183.63, 181.89, 172.70]
genet_real_50percent_ethernet_rewards = [116, 173]

genet_guided_real_5percent_ethernet_rewards = [237.4, 220.0, 222.6]
print(np.mean(genet_ethernet_rewards))
print(np.mean(genet_guided_real_5percent_ethernet_rewards))
print(np.mean(udr3_ethernet_rewards))
print(np.mean(real_default_ethernet_rewards))

bbr_cellular_rewards = [118.07]
copa_cellular_rewards = [255.84]
cubic_cellular_rewards = [69.75]
vivace_cellular_rewards = [-404]
genet_cellular_rewards = [259.74]
udr1_cellular_rewards = [142.31]
udr2_cellular_rewards = [187.61]
udr3_cellular_rewards = [203.96]
real_default_cellular_rewards = [195.42]


plt.figure()


plt.bar([1], [np.mean(bbr_ethernet_rewards)])
plt.bar([2.5, 3.5, 4.5, 5.5, 6.5],
        [np.mean(udr1_ethernet_rewards), np.mean(udr2_ethernet_rewards),
         np.mean(udr3_ethernet_rewards),
         np.mean(real_default_ethernet_rewards),
         np.mean(udr3_real_5percent_ethernet_rewards)],
        yerr=[compute_std_of_mean(udr1_ethernet_rewards),
              compute_std_of_mean(udr2_ethernet_rewards),
              compute_std_of_mean(udr3_ethernet_rewards),
              compute_std_of_mean(real_default_ethernet_rewards),
              compute_std_of_mean(udr3_real_5percent_ethernet_rewards)])

plt.bar([8, 9],
        [np.mean(cl1_ethernet_rewards), np.mean(cl2_ethernet_rewards)],
        yerr=[compute_std_of_mean(cl1_ethernet_rewards),
              compute_std_of_mean(cl2_ethernet_rewards)])
plt.bar([10.5], [np.mean(genet_ethernet_rewards)],
        yerr=[compute_std_of_mean(genet_ethernet_rewards)])
plt.bar([12, 13, 14, 15, 16, 17], [np.mean(genet_real_1percent_ethernet_rewards),
    np.mean(genet_real_4percent_ethernet_rewards),
    np.mean(genet_real_5percent_ethernet_rewards),
    np.mean(genet_real_6percent_ethernet_rewards),
    np.mean(genet_real_20percent_ethernet_rewards),
    np.mean(genet_real_50percent_ethernet_rewards)],
        yerr=[compute_std_of_mean(genet_real_1percent_ethernet_rewards),
              compute_std_of_mean(genet_real_4percent_ethernet_rewards),
              compute_std_of_mean(genet_real_5percent_ethernet_rewards),
              compute_std_of_mean(genet_real_6percent_ethernet_rewards),
              compute_std_of_mean(genet_real_20percent_ethernet_rewards),
              compute_std_of_mean(genet_real_50percent_ethernet_rewards)])

plt.bar([18.5], [np.mean(genet_guided_real_5percent_ethernet_rewards)],
        yerr=[compute_std_of_mean(genet_guided_real_5percent_ethernet_rewards)])
# plt.title('Ethernet')

ax = plt.gca()
ax.set_xticks([1, 2.5, 3.5, 4.5, 5.5, 6.5, 8, 9, 10.5, 12, 13, 14, 15, 16, 17,
               18.5])
ax.set_xticklabels(['bbr','udr1','udr2','udr3', 'udr_real', 'udr3+\n5%r',
                    'cl1', 'cl2', 'genet', 'g+\n1%r', 'g+\n4%r', 'g+\n5%r',
                    'g+\n6%r', 'g+\n20%r', 'g+\n50%r', 'g+\ng5%r'])
ax.set_ylabel('Reward')


plt.figure()
ax = plt.gca()
plt.bar([1, 2], [np.mean(bbr_ethernet_rewards), np.mean(cubic_ethernet_rewards)])
plt.bar([3.5, 4.5, 5.5, 6.5, 7.5],
        [np.mean(udr3_real_5percent_ethernet_rewards),
         np.mean(udr3_real_10percent_ethernet_rewards),
         np.mean(udr3_real_20percent_ethernet_rewards),
         np.mean(udr3_real_50percent_ethernet_rewards),
         np.mean(real_default_ethernet_rewards)],
        yerr=[compute_std_of_mean(udr3_real_5percent_ethernet_rewards),
              compute_std_of_mean(udr3_real_10percent_ethernet_rewards),
              compute_std_of_mean(udr3_real_20percent_ethernet_rewards),
              compute_std_of_mean(udr3_real_50percent_ethernet_rewards),
              compute_std_of_mean(real_default_ethernet_rewards)])

plt.bar([9], [np.mean(genet_guided_real_5percent_ethernet_rewards)],
        yerr=[compute_std_of_mean(genet_guided_real_5percent_ethernet_rewards)])
# plt.title('Ethernet')

ax = plt.gca()
ax.set_xticks([1, 2, 3.5, 4.5, 5.5, 6.5, 7.5, 9])
# ax.set_xticklabels(['BBR', 'Cubic', 'RL\n(synthetic+real)', 'RL\n(real only)',
#                     'Genet\n(synthetic+real)'], rotation=30)
ax.set_xticklabels(['BBR', 'Cubic', 'RL (synthetic+5%real)', 'RL (synthetic+10%real)',
                    'RL (synthetic+20%real)', 'RL (synthetic+50%real)', 'RL (real only)',
                    'Genet (synthetic+real)'], rotation=30, ha='right', rotation_mode='anchor')
ax.set_ylabel('Test reward')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, 'evalulation_asymptotic_real.pdf'),  bbox_inches='tight')

plt.figure()
ax = plt.gca()
plt.bar([1, 2, 3, 4], [np.mean(bbr_cellular_rewards), np.mean(copa_cellular_rewards),
    np.mean(cubic_cellular_rewards), np.mean(vivace_cellular_rewards)])
plt.bar([5.5, 6.5, 7.5, 8.5],
        [np.mean(udr1_cellular_rewards),
         np.mean(udr2_cellular_rewards),
         np.mean(udr3_cellular_rewards),
         np.mean(real_default_cellular_rewards)],
        yerr=[23.78, 5.03, 4.05, 6.70])
        # yerr=[compute_std_of_mean(udr1_cellular_rewards),
        #       compute_std_of_mean(udr2_cellular_rewards),
        #       compute_std_of_mean(udr3_cellular_rewards),
        #       compute_std_of_mean(real_default_cellular_rewards)])

plt.bar([10], [np.mean(genet_cellular_rewards)], yerr=[3.25])
        # yerr=[compute_std_of_mean(genet_cellular_rewards)])
# plt.title('Celluar')
ax = plt.gca()
ax.set_xticks([1, 2, 3, 4, 5.5, 6.5, 7.5, 8.5, 10])
ax.set_xticklabels(['BBR', 'Copa' ,'Cubic', 'Vivace', 'RL-Small', 'RL-Medium', 'RL-Large',
                    'RL-Real', 'Genet'], rotation=30, ha='right', rotation_mode='anchor')
ax.set_ylabel('Test reward')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0,)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, 'evalulation_generalization_test_cellular.pdf'),  bbox_inches='tight')


plt.figure()
ax = plt.gca()
plt.bar([1, 2], [np.mean(bbr_ethernet_rewards), np.mean(cubic_ethernet_rewards)])
plt.bar([3.5, 4.5, 5.5, 6.5],
        [np.mean(udr1_ethernet_rewards),
         np.mean(udr2_ethernet_rewards),
         np.mean(udr3_ethernet_rewards),
         np.mean(real_default_ethernet_rewards)],
        yerr=[compute_std_of_mean(udr1_ethernet_rewards),
              compute_std_of_mean(udr2_ethernet_rewards),
              compute_std_of_mean(udr3_ethernet_rewards),
              compute_std_of_mean(real_default_ethernet_rewards)])

plt.bar([8], [np.mean(genet_ethernet_rewards)],
        yerr=[compute_std_of_mean(genet_ethernet_rewards)])
# plt.title('Ethernet')
ax = plt.gca()
ax.set_xticks([1, 2, 3.5, 4.5, 5.5, 6.5, 8])
ax.set_xticklabels(['BBR', 'Cubic', 'RL-Small', 'RL-Medium', 'RL-Large', 'RL-Real', 'Genet'], rotation=30, ha='right', rotation_mode='anchor')
ax.set_ylabel('Test reward')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, 'evalulation_generalization_test_ethernet.pdf'),  bbox_inches='tight')


plt.figure()
ax = plt.gca()
plt.bar([1, 2, 3],
        [np.mean(cl1_ethernet_rewards), np.mean(cl2_ethernet_rewards),
         np.mean(udr3_ethernet_rewards)],
        yerr=[compute_std_of_mean(cl1_ethernet_rewards),
              compute_std_of_mean(cl2_ethernet_rewards),
              compute_std_of_mean(udr3_ethernet_rewards)])

plt.bar([4.5], [np.mean(genet_ethernet_rewards)],
        yerr=[compute_std_of_mean(genet_ethernet_rewards)])
ax = plt.gca()
ax.set_xticks([1, 2, 3, 4.5])
ax.set_xticklabels(['Inherent metrics', 'Baseline perf', 'Current RL perf', 'Genet'], rotation=30, ha='right', rotation_mode='anchor')
ax.set_ylabel('Test reward')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, 'evalulation_cl_variants.pdf'),  bbox_inches='tight')

# plt.figure()
# plt.bar([1], [np.mean(bbr_cellular_rewards)])
# plt.bar([2.5, 3.5, 4.5, 5.5],
#         [np.mean(udr1_cellular_rewards), np.mean(udr2_cellular_rewards),
#          np.mean(udr3_cellular_rewards),
#          np.mean(real_default_cellular_rewards)],
#         yerr=[compute_std_of_mean(udr1_cellular_rewards),
#               compute_std_of_mean(udr2_cellular_rewards),
#               compute_std_of_mean(udr3_cellular_rewards),
#               compute_std_of_mean(real_default_cellular_rewards)])
#
# plt.bar([7, 8],
#         [np.mean(cl1_cellular_rewards), np.mean(cl2_cellular_rewards)],
#         yerr=[compute_std_of_mean(cl1_cellular_rewards),
#               compute_std_of_mean(cl2_cellular_rewards)])
# plt.bar([9.5], [np.mean(genet_cellular_rewards)],
#         yerr= [compute_std_of_mean(genet_cellular_rewards)])
# plt.title('Cellular')
#
# ax = plt.gca()
# ax.set_xticks([1, 2.5, 3.5, 4.5, 5.5, 7, 8, 9.5])
# ax.set_xticklabels(['bbr','udr1','udr2','udr3', 'udr_real', 'cl1', 'cl2',
#                     'genet'])


plt.figure()
ax = plt.gca()
# plt.bar([1, 2], [np.mean(bbr_ethernet_rewards), np.mean(copa_ethernet_rewards)])
plt.bar([1, 2, 3], [193, 183, np.mean(cubic_ethernet_rewards)])
plt.bar([4.5, 5.5, 6.5, 7.5],
        [120,
         125,
         104,
         np.mean(real_default_ethernet_rewards)],
        yerr=[compute_std_of_mean(udr1_ethernet_rewards),
              # compute_std_of_mean(udr2_ethernet_rewards),
              10,
              compute_std_of_mean(udr3_ethernet_rewards),
              compute_std_of_mean(real_default_ethernet_rewards)])

plt.bar([9], [215],
        yerr=[compute_std_of_mean(genet_ethernet_rewards)])
# plt.title('Ethernet')
ax = plt.gca()
ax.set_xticks([1, 2, 3, 3.5, 4.5, 5.5, 7.5, 9])
ax.set_xticklabels(['BBR', 'Copa', 'Cubic', 'RL-Small', 'RL-Medium', 'RL-Large', 'RL-Real', 'Genet'], rotation=30, ha='right', rotation_mode='anchor')
ax.set_ylabel('Test reward')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# plt.savefig(os.path.join(SAVE_ROOT, 'evalulation_generalization_test_ethernet.pdf'),  bbox_inches='tight')

plt.show()
