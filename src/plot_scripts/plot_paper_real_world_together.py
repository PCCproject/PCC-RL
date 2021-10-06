import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('seaborn-deep')
# ylab.rcParams['xtick.major.pad']='8'
plt.rcParams['ytick.major.pad']='0.5'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['legend.columnspacing'] = 0.5

# gridspec inside gridspec
fig = plt.figure(figsize=(20, 3), constrained_layout=False)
fig.subplots_adjust(top=0.8)

gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[19, 15], wspace=0.14)

gs0 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0], wspace=0.4)

ax0 = fig.add_subplot(gs0[0])

RobustMPC = [3.664]
FastMPC = [3.58]
BBA=[3.57]
ADR = [3.68]
UDR_1 = [2.367]
UDR_2 = [3.28]
UDR_3 = [2.824]


RobustMPC_err = [0.11]
FastMPC_err = [0.13]
BBA_err=[0.17]
ADR_err = [0.14]
UDR_1_err = [0.17]
UDR_2_err = [0.19]
UDR_3_err = [0.28]

width = 1 #0.8
x = [5, 8, 10]
x_loc = [5, 6.2, 7.8]

ax0.bar( x_loc[0] ,BBA ,yerr=BBA_err ,width=width ,color='C0' ,hatch='' ,alpha=1 ,label="BBA" )
ax0.bar( x_loc[1] ,RobustMPC ,yerr=RobustMPC_err ,width=width ,color='C0' ,hatch='/' ,alpha=1 ,label="RobustMPC" )
ax0.bar( x_loc[2] ,ADR ,yerr=ADR_err ,width=width ,color='C2' ,label="GENET" )
ax0.set_xticks( []  )
ax0.spines['right'].set_visible( False )
ax0.spines['top'].set_visible( False )
ax0.set_title( "Wired" )
ax0.set_ylabel( 'Test reward' )
#ax1.bar( 5 ,FastMPC ,yerr=FastMPC_err ,width=1 ,color='C0' ,hatch='..' ,alpha=1 ,label="MPC" )

ax1 = fig.add_subplot(gs0[1])
RobustMPC = [2.72]
FastMPC = [2.51]
BBA = [2.57]
ADR = [2.96]
UDR_1 = [1.35]
UDR_2 = [1.98]
UDR_3 = [1.64]

RobustMPC_err = [0.14]
FastMPC_err = [0.15]
BBA_err = [0.16]
ADR_err = [0.17]
UDR_1_err = [0.18]
UDR_2_err = [0.31]
UDR_3_err = [0.27]

ax1.bar( x_loc[0] ,BBA ,yerr=BBA_err ,width=width ,color='C0' ,hatch='' ,alpha=1 ,label="BBA" )
ax1.bar( x_loc[1] ,RobustMPC ,yerr=RobustMPC_err ,width=width ,color='C0' ,hatch='/' ,alpha=1 ,label="RobustMPC" )
ax1.bar( x_loc[2] ,ADR ,yerr=ADR_err ,width=width ,color='C2' ,label="GENET" )
ax1.set_xticks( []  )
ax1.spines['right'].set_visible( False )
ax1.spines['top'].set_visible( False )
ax1.set_title( "WiFi" )




ax2 = fig.add_subplot(gs0[2])
RobustMPC = [1.59]
FastMPC = [1.27]
BBA = [1.37]
ADR = [1.85]
UDR_1 = [0.75]
UDR_2 = [1.04]
UDR_3 = [0.824]

RobustMPC_err = [0.22]
FastMPC_err = [0.21]
BBA_err = [0.13]
ADR_err = [0.18]
UDR_1_err = [0.18]
UDR_2_err = [0.11]
UDR_3_err = [0.17]

ax2.bar( x_loc[0] ,BBA ,yerr=BBA_err ,width=width ,color='C0' ,hatch='' ,alpha=1 ,label="BBA" )
ax2.bar( x_loc[1] ,RobustMPC ,yerr=RobustMPC_err ,width=width ,color='C0' ,hatch='/' ,alpha=1 ,label="RobustMPC" )
ax2.bar( x_loc[2] ,ADR ,yerr=ADR_err ,width=width ,color='C2' ,label="GENET" )

ax2.set_xticks( []  )
ax2.spines['right'].set_visible( False )
ax2.spines['top'].set_visible( False )
ax2.set_title( "Cellular" )



ax3 = fig.add_subplot(gs0[3])
RobustMPC = [2.624]
FastMPC = [1.98]
BBA=[2.57]
ADR = [3.208]
UDR_1 = [2.567]
UDR_2 = [2.748]
UDR_3 = [2.424]


RobustMPC_err = [0.41]
FastMPC_err = [0.43]
BBA_err=[0.57]
ADR_err = [0.64]
UDR_1_err = [0.77]
UDR_2_err = [0.79]
UDR_3_err = [0.58]

ax3.bar( x_loc[0] ,BBA ,yerr=BBA_err ,width=width ,color='C0' ,hatch='' ,alpha=1 ,label="BBA" )
ax3.bar( x_loc[1] ,RobustMPC ,yerr=RobustMPC_err ,width=width ,color='C0' ,hatch='/' ,alpha=1 ,label="RobustMPC" )
ax3.bar( x_loc[2] ,ADR ,yerr=ADR_err ,width=width ,color='C2' ,label="GENET" )

ax3.set_xticks( [] )
ax3.spines['right'].set_visible( False )
ax3.spines['top'].set_visible( False )
ax3.set_title( "WiFi 2" )


ax4 = fig.add_subplot(gs0[4])
RobustMPC = [1.99]
FastMPC = [1.87]
BBA = [1.587]
ADR = [2.308]
UDR_1 = [1.87]
UDR_2 = [1.648]
UDR_3 = [1.624]

RobustMPC_err = [0.42]
FastMPC_err = [0.41]
BBA_err = [0.13]
ADR_err = [0.38]
UDR_1_err = [0.28]
UDR_2_err = [0.31]
UDR_3_err = [0.37]

bba_bar = ax4.bar( x_loc[0] ,BBA ,yerr=BBA_err ,width=width ,color='C0' ,hatch='' ,alpha=1 ,label="BBA" )
mpc_bar = ax4.bar( x_loc[1] ,RobustMPC ,yerr=RobustMPC_err ,width=width ,color='C0' ,hatch='/' ,alpha=1 ,label="RobustMPC" )
adr_bar = ax4.bar( x_loc[2] ,ADR ,yerr=ADR_err ,width=width ,color='C2' ,label="GENET" )

ax4.set_xticks([])
ax4.spines['right'].set_visible( False )
ax4.spines['top'].set_visible( False )
ax4.set_title( "WiFi 3" )


ax0.set_ylim(bottom=1.1)
ax1.set_ylim(bottom=1.1)
ax2.set_ylim(bottom=1.1)
ax3.set_ylim(bottom=1.1)
ax4.set_ylim(bottom=1.1)

line_labels = ["BBA", "MPC", "GENET"]
abr_lgd = fig.legend(handles=bba_bar + mpc_bar + adr_bar,     # The line objects
           labels=line_labels,   # The labels for each line
           bbox_to_anchor=(0.33, 1.03),
           loc="upper center",   # Position of legend
           borderpad=0.2,
           ncol=3,
           # handlelength=5
           )



rule_based_xpos = [5, 6.2]

udr_xpos = [1.5, 2, 2.5]
genet_xpos = [7.8, 9]
xlabel_pos = [0.25, 1.75]
xtick_labels = ["Rule-based", "Genet"]
width = 1 #0.8
rule_based_labels = ["BBR", "Cubic"]
udr_labels = ["UDR-1", "UDR-2", "UDR-3"]
genet_labels = ["GENET-BBR", "GENET-Cubic"]
# the following syntax does the same as the GridSpecFromSubplotSpec call above:
gs1 = gs[1].subgridspec(1, 3, wspace=0.4)

ax5 = fig.add_subplot(gs1[0])
rule_based_rewards = [-39.42, 110]
udr_rewards = [99.37, 125.197557, 140]
genet_bbr_reward = 140
genet_cubic_reward = 159
genet_rewards = [genet_bbr_reward, genet_cubic_reward]
rule_based_reward_errs = [6.984868842, 7.615547445]
udr_reward_errs = [17.07, 5.191554036, 9.649202373]
genet_reward_errs = [7.05, 8.95]

rule_based_bars = ax5.bar(rule_based_xpos, rule_based_rewards, yerr=rule_based_reward_errs, width=width)
# udr_bars = axes[1].bar(udr_xpos, udr_rewards, yerr=udr_reward_errs, width=width)
genet_bars = ax5.bar(genet_xpos, genet_rewards, yerr=genet_reward_errs, color='C2', width=width)

for bar, pat in zip(rule_based_bars, ('', '/')):
    bar.set_hatch(pat)
# for bar, pat in zip(udr_bars, ('', '/', '.')):
#     bar.set_hatch(pat)
for bar, pat in zip(genet_bars, ('', '/')):
    bar.set_hatch(pat)
ax5.set_xticks([])
# ax6.set_xticks(xlabel_pos)
# ax6.set_xticklabels(xtick_labels, rotation='horizontal')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.set_title('Wired')
ax5.set_ylabel( 'Test reward' )






ax6 = fig.add_subplot(gs1[1])
rule_based_rewards = [-896.92, -808.66]
udr_rewards = [-579.5319291, -371.7235704, -445.5791161]
genet_bbr_reward = -415.9023823
genet_cubic_reward = -487.1606699
genet_rewards = [genet_bbr_reward, genet_cubic_reward]
rule_based_reward_errs = [113.2364857, 125.64]
udr_reward_errs = [116.8319582, 92.27681657, 69.41868345]
genet_reward_errs = [51.29362928, 85.71123066 ]


rule_based_bars = ax6.bar(rule_based_xpos, rule_based_rewards, yerr=rule_based_reward_errs, width=width)
# udr_bars = axes[0].bar(udr_xpos, udr_rewards, yerr=udr_reward_errs, width=width)
genet_bars = ax6.bar(genet_xpos, genet_rewards, yerr=genet_reward_errs, color='C2', width=width)

for bar, pat in zip(rule_based_bars, ('', '/')):
    bar.set_hatch(pat)
# for bar, pat in zip(udr_bars, ('', '/', '.')):
#     bar.set_hatch(pat)
for bar, pat in zip(genet_bars, ('', '/')):
    bar.set_hatch(pat)
ax6.set_xticks([])
# ax5.set_xticks(xlabel_pos)
# ax5.set_xticklabels(xtick_labels, rotation='horizontal')
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.set_title('WiFi')




ax7 = fig.add_subplot(gs1[2])
rule_based_rewards = [-1721.729242, -4273.316741]
udr_rewards = [-1207.16295102302, -1268.9468709186754, -1507.7577581849391]
genet_cubic_reward = -3178.111251
genet_bbr_reward = -3578.505185
genet_rewards = [genet_bbr_reward, genet_cubic_reward]
rule_based_reward_errs = [179.9800978, 150.2627965]
udr_reward_errs = [113, 219, 381]
genet_reward_errs = [797.7498305, 452.2854324]

# ('baseline rewards:', [-1370.5012566266457, -632.5889687764335, -2227.0486933389534, -10704.768490542292, -8980.91984759647])
# ('udr rewards', [-1207.16295102302, -1268.9468709186754, -1507.7577581849391])
# ('genet_rewards', [-801.003684139103, -1382.0921326483844])
# [157.88247787175098, 19.629270689581574, 359.6993315235786, 683.5831268269818, 1064.3376101599108]
# [113.69592973558912, 219.6974578951898, 381.09180696880026]
# [30.291204928093194, 135.22598608212303]
rule_based_bars = ax7.bar(
    rule_based_xpos, rule_based_rewards, yerr=rule_based_reward_errs, width=width)
# udr_bars = axes[2].bar(udr_xpos, udr_rewards, yerr=udr_reward_errs, width=width)
genet_bars = ax7.bar(genet_xpos, genet_rewards, yerr=genet_reward_errs, color='C2', width=width)

for bar, pat in zip(rule_based_bars, ('', '/')):
    bar.set_hatch(pat)
# for bar, pat in zip(udr_bars, ('', '/', '.')):
#     bar.set_hatch(pat)
for bar, pat in zip(genet_bars, ('', '/')):
    bar.set_hatch(pat)
ax7.set_xticks([])
# ax7.set_xticks(xlabel_pos)
# ax7.set_xticklabels(xtick_labels, rotation='horizontal')
ax7.set_title( "Cellular" )
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
# ax7.set_ylim(top=-500)

line_labels = rule_based_labels + genet_labels
lgd = fig.legend(handles=rule_based_bars + genet_bars,     # The line objects
           labels=line_labels,   # The labels for each line
           # bbox_to_anchor=(-0.7, 0.9),
           bbox_to_anchor=(0.73, 1.03),
           # loc="upper right",   # Position of legend
           # loc="lower left",   # Position of legend
           # loc='best',
           loc='upper center',
           borderpad=0.2,
           ncol=4,
           # handlelength=5
           )
# print(lgd.loc)
fig.gca().add_artist(abr_lgd)
# plt.suptitle("GridSpec Inside GridSpec")
# format_axes(fig)

plt.savefig('../../figs/real_world_bars_abr_cc.pdf', bbox_inches='tight')
plt.show()
# fig2 = plt.figure(constrained_layout=False)
# spec2 = gridspec.GridSpec(ncols=8, nrows=1, figure=fig2)
# f2_ax1 = fig2.add_subplot(spec2[0])
# f2_ax2 = fig2.add_subplot(spec2[1])
# f2_ax3 = fig2.add_subplot(spec2[2])
# f2_ax4 = fig2.add_subplot(spec2[3])
# f2_ax5 = fig2.add_subplot(spec2[4])
# f2_ax6 = fig2.add_subplot(spec2[5])
# f2_ax7 = fig2.add_subplot(spec2[6])
# f2_ax8 = fig2.add_subplot(spec2[7])


# fig3 = plt.figure(constrained_layout=True)
# gs = fig3.add_gridspec(3, 3)
# f3_ax1 = fig3.add_subplot(gs[0, :])
# f3_ax1.set_title('gs[0, :]')
# f3_ax2 = fig3.add_subplot(gs[1, :-1])
# f3_ax2.set_title('gs[1, :-1]')
# f3_ax3 = fig3.add_subplot(gs[1:, -1])
# f3_ax3.set_title('gs[1:, -1]')
# f3_ax4 = fig3.add_subplot(gs[-1, 0])
# f3_ax4.set_title('gs[-1, 0]')
# f3_ax5 = fig3.add_subplot(gs[-1, -2])
# f3_ax5.set_title('gs[-1, -2]')
