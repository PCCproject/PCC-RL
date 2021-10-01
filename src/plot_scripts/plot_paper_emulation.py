import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['legend.columnspacing'] = 0.5


fig, axes = plt.subplots(1, 2, figsize=(10, 10))

rule_based_rewards = [168.52, 298.24, 249.45, 63.87]
udr_rewards = [288.15, 309.77, 281.57]
genet_bbr_reward = 312.54
genet_cubic_reward = 314.57
genet_rewards = [genet_bbr_reward, genet_cubic_reward]
rule_based_reward_errs = [25.88, 18.84, 22.25, 29.94]
udr_reward_errs = [18.13, 19.41, 21.65]
genet_reward_errs = [21.28, 24.79]


rule_based_xpos = [0, 0.5, 1, 1.5]
udr_xpos = [2.5, 3, 3.5]
genet_xpos = [4.5, 5]
xlabel_pos = [0.75, 3, 4.75]
width = 0.5
rule_based_labels = ["BBR", "Copa", "Cubic", "Vivace-latency"]
udr_labels = ["UDR-1", "UDR-2", "UDR-3"]
genet_labels = ["GENET-Cubic", "GENET-BBR"]

rule_based_bars = axes[0].bar(
    rule_based_xpos, rule_based_rewards, yerr=rule_based_reward_errs, width=width)
udr_bars = axes[0].bar(udr_xpos, udr_rewards, yerr=udr_reward_errs, width=width)
genet_bars = axes[0].bar(genet_xpos, genet_rewards,
                         yerr=genet_reward_errs, width=width)

for bar, pat in zip(rule_based_bars, ('', '/', '.', '-')):
    bar.set_hatch(pat)
for bar, pat in zip(udr_bars, ('', '/', '.')):
    bar.set_hatch(pat)
axes[0].set_xticks(xlabel_pos)
axes[0].set_xticklabels(["Rule-based", "UDR", "Genet"], rotation='horizontal')


rule_based_rewards = [185.82, 215.43, 236.18, 32.83]
udr_rewards = [156.67, 187.25, 255.45]
genet_bbr_reward = 277.52
genet_cubic_reward = 267.61
genet_rewards = [genet_bbr_reward, genet_cubic_reward]
rule_based_reward_errs = [18.56, 29.95, 14.26, 31.23]
udr_reward_errs = [14.03, 11.37, 14.22]
genet_reward_errs = [13.72, 14.98]

rule_based_bars = axes[1].bar(
    rule_based_xpos, rule_based_rewards, yerr=rule_based_reward_errs, width=width)
udr_bars = axes[1].bar(udr_xpos, udr_rewards, yerr=udr_reward_errs, width=width)
genet_bars = axes[1].bar(genet_xpos, genet_rewards, yerr=genet_reward_errs, width=width)

for bar, pat in zip(rule_based_bars, ('', '/', '.', '-')):
    bar.set_hatch(pat)
for bar, pat in zip(udr_bars, ('', '/', '.')):
    bar.set_hatch(pat)
axes[1].set_xticks(xlabel_pos)
axes[1].set_xticklabels(["Rule-based", "UDR", "Genet"], rotation='horizontal')


lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
print(lines_labels)
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
line_labels = rule_based_labels + udr_labels + genet_labels

print(type(rule_based_bars))

fig.legend(handles=rule_based_bars + udr_bars + genet_bars,     # The line objects
           labels=line_labels,   # The labels for each line
           bbox_to_anchor=(0.5, 1),
           loc="upper center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           ncol=5,
           handlelength=2)
# plt.legend(rule_based_bars + udr_bars + genet_bars,
#            rule_based_labels + udr_labels + genet_labels,
#            bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", ncol=4,
#            mode='expand')
plt.savefig('../../figs/emulation_bars.pdf')
# plt.show()
