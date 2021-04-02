import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# df = pd.read_csv("./tmp/sim_debug_log")

# mask = df['event_type'] == 'S'

# plt.plot(df[mask]['event_time'], np.arange(len(df[mask]['event_time'])))
# print(df)
emu = pd.read_csv("./tmp/obs.csv", header=None)
sim = pd.read_csv("./obs.csv", header = None)
plt.plot(np.arange(len(emu[0])), emu[0], "-x", label='Emulation')
plt.plot(np.arange(len(sim[0])), sim[0], "-x", label='Simulation')
plt.legend()

plt.show()
