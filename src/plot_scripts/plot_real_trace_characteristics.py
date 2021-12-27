import os

import matplotlib.pyplot as plt

from simulator.trace import Trace


TRACE_ROOT = "../../data/2019-09-17T22-29-AWS-California-1-to-Stanford-cellular-3-runs"
TARGET_CCS = ["bbr", "cubic", "vegas", "indigo", "ledbat", "quic"]

def main():
    for cc in TARGET_CCS:
        trace = Trace.load_from_pantheon_file(os.path.join(
            TRACE_ROOT, "{}_datalink_run1.log".format(cc)), loss=0, queue=500)
        print("{}, {}ms, {}Hz".format(trace.avg_bw, trace.avg_delay, trace.bw_change_freq))
        # plt.plot(trace.timestamps, trace.bandwidths)
        break
    # plt.savefig("curve.png")


if __name__ == "__main__":
    main()
