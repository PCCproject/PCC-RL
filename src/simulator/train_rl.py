import argparse
import os
import time
import warnings

from mpi4py.MPI import COMM_WORLD

from simulator.aurora import Aurora
from common.utils import set_seed

warnings.filterwarnings("ignore")


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Training code.")
    parser.add_argument('--exp-name', type=str, default="",
                        help="Experiment name.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to save the model.")
    # parser.add_argument('--gamma', type=float, default=0.99, help='gamma.')

    parser.add_argument('--seed', type=int, default=20, help='seed')
    parser.add_argument("--total-timesteps", type=int, default=100,
                        help="Total number of steps to be trained.")
    parser.add_argument("--pretrained-model-path", type=str, default=None,
                        help="Path to a pretrained Tensorflow checkpoint!")
    parser.add_argument("--randomization-range-file", type=str, default=None,
                        help="A json file which contains a list of "
                        "randomization ranges with their probabilites.")
    parser.add_argument("--total-trace-count", type=int, default=500)
    parser.add_argument("--duration", type=float, default=10,
                        help='trace duration. Unit: second.')
    parser.add_argument("--tensorboard-log", type=str, default=None,
                        help="tensorboard log direcotry.")
    parser.add_argument('--validation', action='store_true',
                        help='specify to enable validation.')

    return parser.parse_args()


def main():
    args = parse_args()
    assert args.pretrained_model_path is None or args.pretrained_model_path.endswith(
        ".ckpt")
    log_dir = args.save_dir
    os.makedirs(log_dir, exist_ok=True)
    set_seed(args.seed + COMM_WORLD.Get_rank() * 100)
    nprocs = COMM_WORLD.Get_size()

    # Initialize model and agent policy
    aurora = Aurora(args.seed + COMM_WORLD.Get_rank() * 100, args.save_dir,
                    int(7200/ nprocs), args.pretrained_model_path,
                    tensorboard_log=args.tensorboard_log)
    # training_traces, validation_traces,
    aurora.train(args.randomization_range_file,
                 args.total_timesteps, tot_trace_cnt= args.total_trace_count,
                 tb_log_name=args.exp_name, validation_flag=args.validation)


if __name__ == '__main__':
    t_start = time.time()
    main()
