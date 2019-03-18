import json
import matplotlib.pyplot as plt
import numpy as np

x = range(1, 10)
y = []
for i in x:
    data = {}
    with open("pcc_env_log_run_%d.json" % (100 * i)) as f:
        data = json.load(f)

    rews = [float(event["Reward"]) for event in data["Events"]]
    avg_reward = np.mean(rews)
    y.append(avg_reward)

plt.plot(x, y)
plt.savefig("reward_graph.png")
