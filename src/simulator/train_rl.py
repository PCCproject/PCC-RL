import argparse
import itertools
import os
import time
import warnings

import ipdb
import numpy as np

# from simulator import network_sim
# from simulator.network_simulator import network
# from simulator import good_network_sim
from simulator import network
from simulator.trace import generate_trace, generate_traces
from simulator.aurora import Aurora

warnings.filterwarnings("ignore")


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Training code.")
    parser.add_argument('--exp-name', type=str, default="",
                        help="Experiment name.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to save the model.")
    # parser.add_argument('--gamma', type=float, default=0.99, help='gamma.')
    parser.add_argument("--delay", type=float,  nargs=2, default=[
                        0.05, 0.05], help="delay range. Unit: millisecond. "
                        "This range will not be used if configuration file is "
                        "provided.")
    parser.add_argument("--bandwidth", type=float, nargs=2, default=[2, 2],
                        help="Bandwidth range. Unit: Mbps. This range will "
                        "not be used if configuration file is provided.")
    parser.add_argument("--loss", type=float, nargs=2, default=[0, 0],
                        help="Loss range. This range will not be used if "
                        "configuration file is provided.")
    parser.add_argument("--queue", type=float, nargs=2, default=[100, 100],
                        help="Queue size range. Unit: packet. This range will "
                        "not be used if configuration file is provided.")

    parser.add_argument("--val-delay", type=float, nargs="+", default=[])
    parser.add_argument("--val-bandwidth", type=float,
                        nargs="+", default=[])
    parser.add_argument("--val-loss", type=float, nargs="+", default=[])
    parser.add_argument("--val-queue", type=float, nargs="+", default=[])

    # parser.add_argument('--arch', type=str, default="32,16", help='arch.')
    parser.add_argument('--seed', type=int, default=20, help='seed')
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="Total number of steps to be trained.")
    parser.add_argument("--pretrained-model-path", type=str, default=None,
                        help="Path to a pretrained Tensorflow checkpoint!")
    parser.add_argument("--randomization-range-file", type=str, default=None,
                        help="A json file which contains a list of "
                        "randomization ranges with their probabilites.")
    parser.add_argument("--total-trace-count", type=int, default=100000)
    parser.add_argument("--duration", type=float, default=10,
                        help='trace duration. Unit: second.')
    parser.add_argument("--tensorboard-log", type=str, default=None,
                        help="tensorboard log direcotry.")

    return parser.parse_args()


def check_args(args):
    """Check arg validity."""
    assert args.delay[0] <= args.delay[1]
    assert args.bandwidth[0] <= args.bandwidth[1]
    assert args.loss[0] <= args.loss[1]
    assert args.queue[0] <= args.queue[1]
    assert args.pretrained_model_path is None or args.pretrained_model_path.endswith(
        ".ckpt")


def main():
    args = parse_args()
    check_args(args)
    log_dir = args.save_dir
    os.makedirs(log_dir, exist_ok=True)
    np.random.seed(args.seed)

    if args.randomization_range_file is not None:
        # generate training traces
        training_traces = generate_traces(args.randomization_range_file,
                                          args.total_trace_count,
                                          args.duration)
        # generate validation traces
        validation_traces = generate_traces(
            args.randomization_range_file, 36, args.duration)
    else:
        # generate training traces
        training_traces = [generate_trace(args.duration, args.bandwidth,
                                          args.delay, args.loss, args.queue)
                           for _ in range(args.total_trace_count)]
        # generate validation traces
        validation_traces = [generate_trace(args.duration, (bw, bw),
                                            (lat, lat), (loss, loss),
                                            (queue, queue))
                             for bw, lat, loss, queue in itertools.product(
            args.val_bandwidth, args.val_delay,
            args.val_loss, args.val_queue)]

    # Initialize model and agent policy
    aurora = Aurora(args.seed, args.save_dir, 7200,
                    args.pretrained_model_path,
                    tensorboard_log=args.tensorboard_log)
    aurora.train(training_traces, validation_traces, args.total_timesteps, args.exp_name)

    # with model.graph.as_default():
    #     saver = tf.train.Saver()
    #     saver.save(model.sess, os.path.join(log_dir, "model_to_serve.ckpt"))
    #
    # # Save the model to the location specified below.
    # export_dir = os.path.join(os.path.join(log_dir, "model_to_serve/"))
    # save_model_to_serve(model, export_dir)


if __name__ == '__main__':
    t_start = time.time()
    main()
