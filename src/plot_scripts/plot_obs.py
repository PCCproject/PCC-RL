import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("obs.csv", header=None)
print(df)

fig, axes = plt.subplots(3,1)
axes[0].set_title("")
axes[0].plot(np.arange(len(df[28])), df[28])
axes[0].set_title("")
axes[1].plot(np.arange(len(df[29])), df[29])
axes[2].set_title("send ratio")
axes[2].plot(np.arange(len(df[30])), df[30])



plt.show()
