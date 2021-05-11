import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('test_noise/noise_log.csv', header=None)

plt.title('Change noise every 50ms, noise ~ max(0, N(0.004, std))')
plt.plot(df[0], df[1], 'o-')
plt.xlabel('std (second)')
plt.ylabel('Reward')

plt.savefig('test_noise/noise.png')

