import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('summary.csv')
# mask = (df['bw_avg'] >= 0.5) & (df['bw_avg'] <= 10)
mask = (df['bw_avg'] >= 10) & (df['bw_avg'] <= 100)
plt.scatter(df['change_freq'][mask]*30, df['aurora_reward'][mask] - df['cubic_reward'][mask])
plt.xlabel('change times')
plt.ylabel('aurora - cubic reward')
# plt.title('limit bandwidth to [0.5, 10]')
plt.title('limit bandwidth to [10, 100]')
plt.savefig('relation.png')

# sorted(df['change_freq'])

# plt.show()
