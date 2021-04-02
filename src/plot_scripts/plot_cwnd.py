import pandas as pd
import matplotlib.pyplot as plt
plt.figure()
df = pd.read_csv('cwnd_log', header=None)
plt.plot(df[0], df[1], '-', label='cwnd')
# plt.ylim(0, 10)
plt.plot(df[0], df[2], '-', label='ssthresh')
plt.ylabel("cwnd (Number of packets)")
plt.xlabel("Time(s)")
plt.legend()

# plt.figure()
# df = pd.read_csv('throughput_log', header=None)
# plt.plot(df[0], df[3], '-', label='link bw')
# # plt.ylim(0, 10)
# plt.plot(df[0], df[5], '-', label='throughput')
# plt.ylabel("Packets/second")
# plt.xlabel("Time(s)")
# plt.legend()

plt.show()
