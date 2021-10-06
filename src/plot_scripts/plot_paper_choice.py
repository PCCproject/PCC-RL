import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
# plt.rcParams['font.size'] = 20
# plt.rcParams['axes.labelsize'] = 20
# plt.rcParams['legend.fontsize'] = 20
# plt.rcParams['legend.columnspacing'] = 0.5
# plt.rcParams['legend.labelspacing'] = 0.02
# plt.rcParams['figure.figsize'] = (11, 9)

plt.rcParams['ytick.major.pad']=0.5
# plt.rcParams['axes.labelpad']=5
plt.rcParams['font.size'] = 38
plt.rcParams['axes.labelsize'] = 38
plt.rcParams['legend.fontsize'] = 38

#plt.rcParams["figure.figsize"] = (17,5)

RESULT_ROOT = "../../results"

EXP_NAME = "sim_eval_reproduce"
EXP_NAME1 = "sim_eval_bbr_reproduce"
fig_size = (7.5, 5)
# fig_size = (6, 4)
fig, ax = plt.subplots(1, 1, figsize=fig_size)
df = pd.read_csv(os.path.join(RESULT_ROOT, EXP_NAME, "sim_eval_vary_queue_bbr_with_cubic.csv"))
assert isinstance(df, pd.DataFrame)
vals_to_test = df['queue']
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

bbr_reward_err = bbr_avg_rewards.std() / np.sqrt(10)
cubic_reward_err = cubic_avg_rewards.std()/ np.sqrt(10)
genet_cubic_reward_err = genet_avg_rewards.std()/ np.sqrt(10)
width = 0.7
ax.bar([1, 3.5], [bbr_avg_rewards.mean(), cubic_avg_rewards.mean()],
       yerr=[bbr_reward_err, cubic_reward_err], color='C0', width=width, label='Rule-based baselines')
ax.bar([2, 4.5], [genet_avg_rewards.mean(), 421], yerr=[genet_cubic_reward_err, 2.04], color='C2', width=width, label='GENET')
ax.set_xticks([1.5, 4])
ax.set_xticklabels(["BBR", 'Cubic'], rotation='horizontal')
ax.set_ylabel('Test reward', labelpad = 0.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.legend(ncol=2, bbox_to_anchor=(0.45, 1.3), loc='upper center', handlelength=1, handletextpad=0.5, columnspacing=0.5)

# plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
#                 mode="expand", borderaxespad=0, ncol=3)

# line_labels = ["GENET", "BBR", "TCP Cubic", "UDR-1", "UDR-2", "UDR-3"]

# lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# fig.subplots_adjust(top=0.3)
# fig.legend(handles=lines,     # The line objects
#            labels=line_labels,   # The labels for each line
#            bbox_to_anchor=(0.5, 1),
#            loc="upper center",   # Position of legend
#            borderaxespad=0.1,    # Small spacing around legend box
#             ncol=3,
#             handlelength=2)
plt.tight_layout()
# save_dir = os.path.join(RESULT_ROOT, EXP_NAME, "sim_eval_vary_summary.png")
save_dir = os.path.join("../../figs/cc_choice_bars.pdf")
plt.savefig(save_dir, bbox_inches='tight')



fig, ax = plt.subplots(1, 1, figsize=fig_size)
MPC = [1.88]
BO_MPC = [2.27]
BBA = [1.65]
BO_BBA = [2.23]

MPC_err = [0.15]
BO_MPC_err = [0.17]
BBA_err = [0.23]
BO_BBA_err = [0.21]

ax.bar( 1 ,MPC ,yerr=MPC_err ,width=width ,color='C0' )
ax.bar( 2 ,BO_MPC ,yerr=BO_MPC_err ,width=width ,color='C2', alpha=1)
ax.bar( 3.5 ,BBA ,yerr=BBA_err ,width=width ,color='C0', label='Rule-based baselines')
ax.bar( 4.5 ,BO_BBA ,yerr=BO_BBA_err ,width=width ,color='C2', alpha=1, label='GENET'  )

labels = ['MPC', 'BBA']
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1.5, 4])
ax.set_xticklabels(labels, rotation='horizontal')
ax.set_ylabel('Test reward', labelpad = 0.1)
# plt.legend(ncol=2, bbox_to_anchor=(0.45, 1.3), loc='upper center', handlelength=1, handletextpad=0.5, columnspacing=0.5)
plt.tight_layout()
save_dir = os.path.join("../../figs/abr_choice_bars.pdf")
plt.savefig(save_dir, bbox_inches='tight')


fig, ax = plt.subplots(1, 1, figsize=fig_size)
LLF = [490.88/100]
BO_LLF = [590.27/100]
c3 = [506.65/100]
BO_c3 = [604.23/100]

LLF_err = [35.15/100]
BO_LLF_err = [57.17/100]
c3_err = [43.23/100 ]
BO_c3_err = [46.21/100]
print(LLF)
import pdb
pdb.set_trace()
ax.bar( 1 ,LLF ,yerr=LLF_err ,width=width, color='C0', bottom=-10)
ax.bar( 2 ,BO_LLF ,yerr=BO_LLF_err, width=width, color='C2' ,alpha=1, bottom=-10  )
ax.bar( 3.5 ,c3 ,yerr=c3_err ,width=width ,color='C0', label='Rule-based baselines',bottom=-10  )
ax.bar( 4.5 ,BO_c3 ,yerr=BO_c3_err ,width=width ,color='C2' ,alpha=1, bottom=-10,  label='GENET'  )
# plt.legend(ncol=2, bbox_to_anchor=(0.45, 1.3), loc='upper center', handlelength=1, handletextpad=0.5, columnspacing=0.5)

# labels = ['LLF', 'GENET+\nLLF', 'C3', 'GENET+\nC3']
labels = ['LLF', 'C3']
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1.5, 4])
ax.set_xticklabels(labels, rotation='horizontal')
ax.set_ylabel('Test reward', labelpad = 0.1)
plt.tight_layout()
save_dir = os.path.join("../../figs/lb_choice_bars.pdf")
plt.savefig(save_dir, bbox_inches='tight')


plt.show()
