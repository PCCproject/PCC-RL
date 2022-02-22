import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulator.trace import Trace
from simulator.pantheon_dataset import PantheonDataset


TRACE_ROOT = "../../data"
SCENES = ['cellular', 'ethernet']

def main():
    fig, axes = plt.subplots(1, 4)
    for scene, color in zip(SCENES, ['C0', 'C1']):
        avg_delays = []
        avg_bw = []
        bw_std = []
        bw_change_freq = []
        if scene == 'ethernet':
            queue = 500
        elif scene == 'cellular':
            queue = 50
        else:
            queue = 100
        dataset = PantheonDataset(TRACE_ROOT, scene)
        link_dirs = glob.glob(os.path.join(TRACE_ROOT, scene, "*/"))
        traces = dataset.get_traces(0, queue)
        for trace in traces:
            avg_delays.append(trace.avg_delay)
            avg_bw.append(trace.max_bw)
            bw_std.append(trace.std_bw)
            bw_change_freq.append(trace.bw_change_freq)

        axes[0].scatter(bw_std, avg_delays, label=scene + ": {}".format(len(avg_delays)), c=color)
        axes[0].set_ylabel('avg delay (s)')
        axes[0].set_xlabel('bw std')
        axes[0].set_ylim(0, 220)
        axes[0].set_xlim(0, 150)

        axes[1].scatter(bw_change_freq, avg_delays, label=scene + ": {}".format(len(avg_delays)), c=color)
        axes[1].set_ylabel('avg delay (s)')
        axes[1].set_xlabel('bw change freq (Hz)')
        axes[1].set_ylim(0, 220)

        axes[2].scatter(bw_change_freq, bw_std, label=scene + ": {}".format(len(avg_delays)), c=color)
        axes[2].set_ylabel('bw std')
        axes[2].set_xlabel('bw change freq (Hz)')
        axes[2].set_ylim(0, 140)

        axes[3].scatter(avg_bw, avg_delays, label=scene + ": {}".format(len(avg_delays)), c=color)
        axes[3].set_ylabel('avg delay (s)')
        axes[3].set_xlabel('avg bw (Mbps)')
    plt.legend()
    plt.tight_layout()
    # plt.savefig("old_real_trace_scatter.png")
    plt.savefig("real_trace_scatter.png")


if __name__ == "__main__":
    main()
