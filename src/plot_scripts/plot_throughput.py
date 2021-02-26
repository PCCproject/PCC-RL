import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# plt.figure()
# plt.ylim(0, 10)
rl = pd.read_csv('rl_throughput_log', header=None)
cubic = pd.read_csv('cubic_throughput_log', header=None)
plt.figure()
plt.plot(rl[0], rl[6], '-', label='rl')
plt.plot(cubic[0], cubic[6], '-', label='cubic')
plt.ylabel('rewards')
plt.xlabel('time(s)')
plt.legend()
print(np.mean(rl[6]))
print(np.mean(cubic[6]))

plt.figure()
plt.plot(rl[0], rl[5], '-', label='rl')
plt.plot(cubic[0], cubic[5], '-', label='cubic')
plt.ylabel('throughput(pkt/sec)')
plt.xlabel('time(s)')
plt.legend()
plt.show()
