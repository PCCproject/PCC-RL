import json
import matplotlib.pyplot as plt
import numpy as np
import sys

data = {}
with open(sys.argv[1]) as f:
    data = json.load(f)

time_data = [float(event["Time"]) for event in data["Events"][1:]]
rew_data = [float(event["Reward"]) for event in data["Events"][1:]]
send_data = [2.0 * float(event["Send Rate"]) for event in data["Events"][1:]]
thpt_data = [float(event["Throughput"]) for event in data["Events"][1:]]
latency_data = [float(event["Latency"]) for event in data["Events"][1:]]
loss_data = [float(event["Loss Rate"]) for event in data["Events"][1:]]

fig, axes = plt.subplots(5, figsize=(10, 12))
rew_axis = axes[0]
send_axis = axes[1]
thpt_axis = axes[2]
latency_axis = axes[3]
loss_axis = axes[4]

rew_axis.plot(time_data, rew_data)
rew_axis.set_ylabel("Reward")

send_axis.plot(time_data, send_data)
send_axis.set_ylabel("Send Rate")

thpt_axis.plot(time_data, thpt_data)
thpt_axis.set_ylabel("Throughput")

latency_axis.plot(time_data, latency_data)
latency_axis.set_ylabel("Latency")

loss_axis.plot(time_data, loss_data)
loss_axis.set_ylabel("Loss Rate")
loss_axis.set_xlabel("Monitor Interval")

fig.suptitle("Summary Graph for %s" % sys.argv[1])
fig.savefig("output.pdf")
