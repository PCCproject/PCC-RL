import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulator.network_simulator.bbr import BBR
from simulator.network_simulator.cubic import Cubic
from simulator.trace import Trace


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Plot validation curv.")
    parser.add_argument('--log-file', type=str, nargs="+", required=True,
                        help="path to a testing log file.")
    parser.add_argument('--save-dir', type=str, default=None,
                        help="path to save.")

    args, unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    bbr = BBR(False)
    cubic = Cubic(False)

    validation_traces = []
    save_dirs = []
    for i in range(20):
        trace_file = os.path.join(args.save_dir, 'validation_traces', "trace_{}.json".format(i))
        if not os.path.exists(trace_file):
            continue
        validation_traces.append(Trace.load_from_file(trace_file))

        save_dir = os.path.join(args.save_dir, 'validation_traces',"trace_{}".format(i))
        os.makedirs(save_dir, exist_ok=True)
        save_dirs.append(save_dir)
    bbr_trace_rewards = bbr.test_on_traces(validation_traces, save_dirs, False)
    cubic_trace_rewards = cubic.test_on_traces(validation_traces, save_dirs, False)
    bbr_rewards = [mi_level_reward for mi_level_reward, _ in bbr_trace_rewards]
    cubic_rewards = [mi_level_reward for mi_level_reward, _ in cubic_trace_rewards]

    for log_file in args.log_file:
        plt.figure()
        model_name = log_file.split('/')[-2]
        plt.title(model_name)
        df = pd.read_csv(log_file, sep='\t')
        best_step = int(df['num_timesteps'][df['mean_validation_reward'].argmax()])
        t_used = df['tot_t_used(min)'][df['mean_validation_reward'].argmax()]
        best_reward = df['mean_validation_reward'].max()
        best_model_path = os.path.join(os.path.dirname(log_file), "model_step_{}.ckpt.meta".format(best_step))

        plt.plot(df['num_timesteps'], df['mean_validation_reward'],
                 'o-', label="best_reward: {:.2f}, best step: {}, used {:.2f}min".format(best_reward, int(best_step), t_used))
        plt.axhline(y=np.mean(bbr_rewards), c='r', label='BBR')
        plt.axhline(y=np.mean(cubic_rewards), c='k', label='Cubic')
        plt.xlabel('Num steps')
        plt.ylabel('Validation Reward')
        plt.legend()
        assert os.path.exists(best_model_path)
        print(best_model_path.replace(".meta", ""))
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            plt.savefig(os.path.join(args.save_dir,
                                     '{}_val_curve.png'.format(model_name)))
        plt.close()


if __name__ == '__main__':
    main()
