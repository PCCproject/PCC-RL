import argparse
import os
import time
import warnings

from mpi4py.MPI import COMM_WORLD

from simulator.aurora import Aurora
from simulator.trace import Trace
from common.utils import set_seed, save_args

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
    parser.add_argument("--train-trace-file", type=str, default=None,
                        help="A file contains a list of paths to the training "
                        "traces.")
    parser.add_argument("--val-trace-file", type=str, default=None,
                        help="A file contains a list of paths to the validation"
                        " traces.")
    parser.add_argument("--total-trace-count", type=int, default=500)
    parser.add_argument("--duration", type=float, default=10,
                        help='trace duration. Unit: second.')
    parser.add_argument("--tensorboard-log", type=str, default=None,
                        help="tensorboard log direcotry.")
    parser.add_argument('--validation', action='store_true',
                        help='specify to enable validation.')
    parser.add_argument('--dataset', type=str, default='pantheon',
                        choices=('pantheon', 'synthetic'), help='dataset name')
    parser.add_argument('--real-trace-prob', type=float, default=0.0,
                        help='Probability of picking a real trace in training')

    return parser.parse_args()


def main():
    args = parse_args()
    assert args.pretrained_model_path is None or args.pretrained_model_path.endswith(
        ".ckpt")
    os.makedirs(args.save_dir, exist_ok=True)
    save_args(args, args.save_dir)
    set_seed(args.seed + COMM_WORLD.Get_rank() * 100)
    nprocs = COMM_WORLD.Get_size()

    # Initialize model and agent policy
    aurora = Aurora(args.seed + COMM_WORLD.Get_rank() * 100, args.save_dir,
                    int(7200 / nprocs), args.pretrained_model_path,
                    tensorboard_log=args.tensorboard_log)
    # training_traces, validation_traces,
    training_traces = []
    val_traces = []
    if args.train_trace_file:
        with open(args.train_trace_file, 'r') as f:
            for line in f:
                line = line.strip()
                training_traces.append(Trace.load_from_file(line))

    if args.val_trace_file:
        with open(args.val_trace_file, 'r') as f:
            for line in f:
                line = line.strip()
                if args.dataset == 'pantheon':
                    queue = 100  # dummy value
                    # if "ethernet" in line:
                    #     queue = 500
                    # elif "cellular" in line:
                    #     queue = 50
                    # else:
                    #     queue = 100
                    val_traces.append(Trace.load_from_pantheon_file(
                        line, queue=queue, loss=0))
                elif args.dataset == 'synthetic':
                    val_traces.append(Trace.load_from_file(line))
                else:
                    raise ValueError

    aurora.train(args.randomization_range_file,
                 args.total_timesteps, tot_trace_cnt=args.total_trace_count,
                 tb_log_name=args.exp_name, validation_flag=args.validation,
                 training_traces=training_traces,
                 validation_traces=val_traces,
                 real_trace_prob=args.real_trace_prob)


if __name__ == '__main__':
    t_start = time.time()
    main()
