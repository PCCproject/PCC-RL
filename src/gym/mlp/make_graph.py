import json
import matplotlib.pyplot as plt
import numpy as np

x = []
y = []
i = 0
done = False
while not done:
    try:
        i += 1
        data = {}
        with open("pcc_env_log_run_%d.json" % (100 * i)) as f:
            data = json.load(f)
        rews = [float(event["Reward"]) for event in data["Events"]]
        avg_reward = np.mean(rews)
        x.append(i)
        y.append(avg_reward)
    except:
        done = True

plt.plot(x, y)
plt.savefig("reward_graph.png")
