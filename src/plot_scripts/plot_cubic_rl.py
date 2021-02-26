import pandas as pd
import matplotlib.pyplot as plt

aurora = pd.read_csv("tmp/rl_test/rl_test_log0.csv")
cubic = pd.read_csv("tmp_cubic/cubic_test_log0.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(aurora['timestamp'],
                aurora['throughput'] * 1500 * 8 / 1000000, c='C0',
                label='{} recv rate, avg {:.3f}mbps'.format(
                    "aurora", aurora['throughput'].mean() * 1500 * 8 / 1000000))
axes[0, 0].plot(aurora['timestamp'],
                aurora['send_throughput'] * 1500 * 8 / 1000000, ls='--',
                c='C0', label='{} send rate, avg {:.3f}mbps'.format(
                    "aurora", aurora['send_throughput'].mean() * 1500 * 8 / 1000000))

axes[0, 0].plot(cubic['timestamp'],
                cubic['throughput'] * 1500 * 8 / 1000000, c='C1',
                label='{} recv rate, avg {:.3f}mbps'.format(
                    "cubic", cubic['throughput'].mean() * 1500 * 8 / 1000000))
axes[0, 0].plot(cubic['timestamp'],
                cubic['send_throughput'] * 1500 * 8 / 1000000, ls='--',
                c='C1',
                label='{} send rate, avg {:.3f}mbps'.format(
                    "cubic", cubic['send_throughput'].mean() * 1500 * 8 / 1000000))
axes[0, 0].plot(cubic['timestamp'], cubic['link0_bw'] * 1500 * 8 / 1000000,
                c='C2', label='bw, avg {}mbps'.format(cubic['link0_bw'].mean() * 1500 * 8 / 1000000))
axes[0, 0].set_title("Simulator, Rate")
axes[0, 0].set_xlabel("Time(s)")
axes[0, 0].set_ylabel("mbps")
axes[0, 0].legend()


axes[0, 1].scatter(aurora['timestamp'], aurora['latency'] * 100, c='C0',
        label='Aurora, RTT, avg {:.2f}ms'.format(aurora['latency'].mean() * 1000))
axes[0, 1].scatter(cubic['timestamp'], cubic['latency'] * 100, c='C1',
        label='Cubic, RTT, avg {:.2f}ms'.format(cubic['latency'].mean() * 1000))
axes[0, 1].set_title("Simulator(Link RTT=100ms), RTT")
axes[0, 1].set_xlabel("Time(s)")
axes[0, 1].set_ylabel("ms")
axes[0, 1].legend()


plt.tight_layout()
plt.show()
