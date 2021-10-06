from common.utils import set_seed
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-deep')
set_seed(10)

df_genet_bbr = pd.read_csv('training_curve_genet_bbr.csv')
df_udr = pd.read_csv('training_curve_udr.csv')
assert isinstance(df_genet_bbr, pd.DataFrame)
assert isinstance(df_udr, pd.DataFrame)
genet_steps = df_genet_bbr['genet_steps'] / 1e3
steps = df_udr['steps'] / 1e3
udr1_avg_rewards = df_udr['udr1_avg_rewards']
udr2_avg_rewards = df_udr['udr2_avg_rewards']
udr3_avg_rewards = df_udr['udr3_avg_rewards']
genet_avg_rewards = df_genet_bbr['genet_avg_rewards']
plt.plot(genet_steps, df_genet_bbr['genet_avg_rewards'], c='r')
genet_reward_errs = np.concatenate([((df_udr['udr1_up_bnd'] - df_udr['udr1_low_bnd']) / 2).to_numpy(), ((df_udr['udr3_up_bnd'] - df_udr['udr3_low_bnd']) / 2).to_numpy()])
print(genet_reward_errs)
genet_reward_errs = genet_reward_errs[:36]
print(len(genet_reward_errs))
assert len(genet_avg_rewards) == len(genet_reward_errs)
genet_low_bnd = genet_avg_rewards.to_numpy() - genet_reward_errs
genet_up_bnd = genet_avg_rewards.to_numpy() + genet_reward_errs
print(genet_up_bnd)
print(genet_low_bnd)
plt.fill_between(genet_steps, np.array(genet_low_bnd), np.array(genet_up_bnd), color='r', alpha=0.1)
udr1_low_bnd = df_udr['udr1_low_bnd']
udr1_up_bnd = df_udr['udr1_up_bnd']
udr2_low_bnd = df_udr['udr2_low_bnd']
udr2_up_bnd = df_udr['udr2_up_bnd']
udr3_low_bnd = df_udr['udr3_low_bnd']
udr3_up_bnd = df_udr['udr3_up_bnd']

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.axhline(y=np.mean(448), ls="--", label="BBR")
plt.axhline(y=np.mean(441), ls="-.", label="Cubic")
plt.plot(steps, udr1_avg_rewards, "-", label='UDR-1')
plt.fill_between(steps, udr1_low_bnd, udr1_up_bnd, color='grey', alpha=0.1)
plt.plot(steps, udr2_avg_rewards, "--", label='UDR-2')
plt.fill_between(steps, udr2_low_bnd, udr2_up_bnd, color='grey', alpha=0.1)
plt.plot(steps, udr3_avg_rewards, "-.", label='UDR-3')
plt.fill_between(steps, udr3_low_bnd, udr3_up_bnd, color='grey', alpha=0.1)
plt.xlim(0, steps.iloc[-1])
# print(steps.to_list())
# print(udr1_avg_rewards.to_list())
# print(udr1_up_bnd.to_list())
# print(udr1_low_bnd.to_list())

# print(udr2_avg_rewards.to_list())
# print(udr2_up_bnd.to_list())
# print(udr2_low_bnd.to_list())

# print(udr3_avg_rewards.to_list())
# print(udr3_up_bnd.to_list())
# print(udr3_low_bnd.to_list())

print(genet_steps.to_list())
print(genet_avg_rewards.to_list())
print(genet_up_bnd.tolist())
print(genet_low_bnd.tolist())

import pdb
pdb.set_trace()

plt.xlabel('Training epochs (1e3)')
plt.ylabel('Test reward')
plt.legend()
plt.savefig('../../figs/training_curves.pdf')
plt.show()
