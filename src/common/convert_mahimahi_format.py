import argparse
import os
import glob

import numpy as np

from simulator.trace import Trace


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Generate trace files.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to save the model.")
    # parser.add_argument('--trace-file', type=str, required=True,
    #                     help='Path to trace file')
    parser.add_argument('--trace-dir', type=str, required=True,
                        help='Path to trace file')
    args, unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    for trace_file in glob.glob(os.path.join(args.trace_dir, "*.json")):
        trace_name = os.path.splitext(os.path.basename(trace_file))[0]
        tr =Trace.load_from_file(trace_file)
        ms_series = tr.convert_to_mahimahi_format()
        with open(os.path.join(args.save_dir, trace_name), 'w', 1) as f:
            for ms in ms_series:
                f.write(str(ms) + '\n')

        with open(os.path.join(args.save_dir, 'loss'), 'w', 1) as f:
            f.write(str(tr.loss_rate))
        with open(os.path.join(args.save_dir, 'queue'), 'w', 1) as f:
            f.write(str(int(tr.queue_size)))
        with open(os.path.join(args.save_dir, 'delay'), 'w', 1) as f:
            f.write(str(int(np.mean(np.array(tr.delays)))))


if __name__ == '__main__':
    main()
