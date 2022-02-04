import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
# plt.rcParams['font.size'] = 2.8
# plt.rcParams['axes.labelsize'] = 2.8
# plt.rcParams['legend.fontsize'] = 2.8
# plt.rcParams['legend.columnspacing'] = 0.5


fig, axes = plt.subplots(1, 3, figsize=(12, 2))


# wifi
rule_based_rewards = [-896.92, -808.66]
udr_rewards = [-579.5319291, -371.7235704, -445.5791161]
genet_bbr_reward = -415.9023823
genet_cubic_reward = -487.1606699
genet_rewards = [genet_bbr_reward, genet_cubic_reward]
rule_based_reward_errs = [113.2364857, 125.64]
udr_reward_errs = [116.8319582, 92.27681657, 69.41868345]
genet_reward_errs = [51.29362928, 85.71123066 ]



rule_based_xpos = [0, 0.5]
udr_xpos = [1.5, 2, 2.5]
genet_xpos = [1.5, 2]
xlabel_pos = [0.25, 1.75]
xtick_labels = ["Rule-based", "Genet"]
width = 0.5
rule_based_labels = ["BBR", "Cubic"]
udr_labels = ["UDR-1", "UDR-2", "UDR-3"]
genet_labels = ["GENET-BBR", "GENET-Cubic"]

rule_based_bars = axes[0].bar(
    rule_based_xpos, rule_based_rewards, yerr=rule_based_reward_errs, width=width)
# udr_bars = axes[0].bar(udr_xpos, udr_rewards, yerr=udr_reward_errs, width=width)
genet_bars = axes[0].bar(genet_xpos, genet_rewards,
                         yerr=genet_reward_errs, color='C2', width=width)

for bar, pat in zip(rule_based_bars, ('', '/', '.', '-')):
    bar.set_hatch(pat)
# for bar, pat in zip(udr_bars, ('', '/', '.')):
#     bar.set_hatch(pat)
for bar, pat in zip(genet_bars, ('', '/')):
    bar.set_hatch(pat)
axes[0].set_xticks(xlabel_pos)
axes[0].set_xticklabels(xtick_labels, rotation='horizontal')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# wired
rule_based_rewards = [-39.42, 110]
udr_rewards = [99.37, 125.197557, 140]
genet_bbr_reward = 140
genet_cubic_reward = 159
genet_rewards = [genet_bbr_reward, genet_cubic_reward]
rule_based_reward_errs = [6.984868842, 7.615547445]
udr_reward_errs = [17.07, 5.191554036, 9.649202373]
genet_reward_errs = [7.05, 8.95]

rule_based_bars = axes[1].bar(
    rule_based_xpos, rule_based_rewards, yerr=rule_based_reward_errs, width=width)
# udr_bars = axes[1].bar(udr_xpos, udr_rewards, yerr=udr_reward_errs, width=width)
genet_bars = axes[1].bar(genet_xpos, genet_rewards, yerr=genet_reward_errs, color='C2', width=width)

for bar, pat in zip(rule_based_bars, ('', '.')):
    bar.set_hatch(pat)
# for bar, pat in zip(udr_bars, ('', '/', '.')):
#     bar.set_hatch(pat)
for bar, pat in zip(genet_bars, ('', '/')):
    bar.set_hatch(pat)
axes[1].set_xticks(xlabel_pos)
axes[1].set_xticklabels(xtick_labels, rotation='horizontal')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)


# cellular
rule_based_rewards = [-1721.729242, -4273.316741]
udr_rewards = [-1207.16295102302, -1268.9468709186754, -1507.7577581849391]
genet_cubic_reward = -3178.111251
genet_bbr_reward = -3578.505185
genet_rewards = [genet_bbr_reward, genet_cubic_reward]
rule_based_reward_errs = [179.9800978, 150.2627965]
udr_reward_errs = [113, 219, 381]
genet_reward_errs = [797.7498305, 452.2854324]


# 2nd rnd measure
# rule_based_rewards = [-1370.5012566266457, -2227.0486933389534]
# udr_rewards = [-1207.16295102302, -1268.9468709186754, -1507.7577581849391]
# genet_cubic_reward = -801.003684139103
# genet_bbr_reward = -1382.0921326483844
# genet_rewards = [genet_bbr_reward, genet_cubic_reward]
# rule_based_reward_errs = [157.88247787175098, 359.6993315235786]
# udr_reward_errs = [113, 219, 381]
# genet_reward_errs = [30, 135]

# ('baseline rewards:', [-1370.5012566266457, -632.5889687764335, -2227.0486933389534, -10704.768490542292, -8980.91984759647])
# ('udr rewards', [-1207.16295102302, -1268.9468709186754, -1507.7577581849391])
# ('genet_rewards', [-801.003684139103, -1382.0921326483844])
# [157.88247787175098, 19.629270689581574, 359.6993315235786, 683.5831268269818, 1064.3376101599108]
# [113.69592973558912, 219.6974578951898, 381.09180696880026]
# [30.291204928093194, 135.22598608212303]
rule_based_bars = axes[2].bar(
    rule_based_xpos, rule_based_rewards, yerr=rule_based_reward_errs, width=width)
# udr_bars = axes[2].bar(udr_xpos, udr_rewards, yerr=udr_reward_errs, width=width)
genet_bars = axes[2].bar(genet_xpos, genet_rewards, yerr=genet_reward_errs, color='C2', width=width)

for bar, pat in zip(rule_based_bars, ('', '.')):
    bar.set_hatch(pat)
# for bar, pat in zip(udr_bars, ('', '/', '.')):
#     bar.set_hatch(pat)
for bar, pat in zip(genet_bars, ('', '/')):
    bar.set_hatch(pat)
axes[2].set_xticks(xlabel_pos)
axes[2].set_xticklabels(xtick_labels, rotation='horizontal')
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)


# lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
# print(lines_labels)
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# line_labels = rule_based_labels + udr_labels + genet_labels
# lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
# print(lines_labels)
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
line_labels = rule_based_labels + genet_labels

print(type(rule_based_bars))

# fig.subplots_adjust(top=0.3)
lgd = fig.legend(handles=rule_based_bars + genet_bars,     # The line objects
           labels=line_labels,   # The labels for each line
           bbox_to_anchor=(0.5, 1.2),
           loc="upper center",   # Position of legend
           # borderpad=2,
           ncol=4,
           # handlelength=5
           )
# fig.tight_layout()
plt.savefig('../../figs/real_world_bars.pdf', bbox_inches='tight')
# plt.show()
