import argparse
import os
import time
import warnings

from mpi4py.MPI import COMM_WORLD

from simulator.network_simulator.pcc.aurora.aurora import Aurora
from simulator.network_simulator.pcc.aurora.schedulers import (
    UDRTrainScheduler,
    CL1TrainScheduler,
    CL2TrainScheduler,
)
from simulator.trace import Trace
from common.utils import set_seed, save_args

warnings.filterwarnings("ignore")


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Training code.")
    parser.add_argument(
        "--exp-name", type=str, default="", help="Experiment name."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="direcotry to save the model.",
    )
    parser.add_argument("--seed", type=int, default=20, help="seed")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100,
        help="Total number of steps to be trained.",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default="",
        help="Path to a pretrained Tensorflow checkpoint!",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default=None,
        help="tensorboard log direcotry.",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="specify to enable validation.",
    )
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="pantheon",
    #     choices=("pantheon", "synthetic"),
    #     help="dataset name",
    # )
    subparsers = parser.add_subparsers(dest="curriculum", help="CL parsers.")
    udr_parser = subparsers.add_parser("udr", help="udr")
    udr_parser.add_argument(
        "--real-trace-prob",
        type=float,
        default=0.0,
        help="Probability of picking a real trace in training",
    )
    udr_parser.add_argument(
        "--train-trace-file",
        type=str,
        default="",
        help="A file contains a list of paths to the training traces.",
    )
    udr_parser.add_argument(
        "--val-trace-file",
        type=str,
        default="",
        help="A file contains a list of paths to the validation traces.",
    )
    udr_parser.add_argument(
        "--config-file",
        type=str,
        default="",
        help="A json file which contains a list of randomization ranges with "
        "their probabilites.",
    )
    cl1_parser = subparsers.add_parser("cl1", help="cl1")
    cl1_parser.add_argument(
        "--config-files",
        type=str,
        nargs="+",
        help="A list of randomization config files.",
    )
    cl2_parser = subparsers.add_parser("cl2", help="cl2")
    cl2_parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        choices=("bbr", "bbr_old", "cubic"),
        help="Baseline used to sort environments.",
    )
    cl2_parser.add_argument(
        "--config-file",
        type=str,
        default="",
        help="A json file which contains a list of randomization ranges with "
        "their probabilites.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    assert (
        not args.pretrained_model_path
        or args.pretrained_model_path.endswith(".ckpt")
    )
    os.makedirs(args.save_dir, exist_ok=True)
    save_args(args, args.save_dir)
    set_seed(args.seed + COMM_WORLD.Get_rank() * 100)
    nprocs = COMM_WORLD.Get_size()

    # Initialize model and agent policy
    aurora = Aurora(
        args.seed + COMM_WORLD.Get_rank() * 100,
        args.save_dir,
        int(7200 / nprocs),
        args.pretrained_model_path,
        tensorboard_log=args.tensorboard_log,
    )
    # training_traces, validation_traces,
    training_traces = []
    val_traces = []
    if args.curriculum == "udr":
        config_file = args.config_file
        if args.train_trace_file:
            with open(args.train_trace_file, "r") as f:
                for line in f:
                    line = line.strip()
                    training_traces.append(Trace.load_from_file(line))

        if args.validation and args.val_trace_file:
            with open(args.val_trace_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if args.dataset == "pantheon":
                        queue = 100  # dummy value
                        val_traces.append(
                            Trace.load_from_pantheon_file(
                                line, queue=queue, loss=0
                            )
                        )
                    elif args.dataset == "synthetic":
                        val_traces.append(Trace.load_from_file(line))
                    else:
                        raise ValueError
        train_scheduler = UDRTrainScheduler(
            config_file,
            training_traces,
            percent=args.real_trace_prob,
        )
    elif args.curriculum == "cl1":
        config_file = args.config_files[0]
        train_scheduler = CL1TrainScheduler(args.config_files, aurora)
    elif args.curriculum == "cl2":
        config_file = args.config_file
        train_scheduler = CL2TrainScheduler(
            config_file, aurora, args.baseline
        )
    else:
        raise NotImplementedError

    aurora.train(
        config_file,
        args.total_timesteps,
        train_scheduler,
        tb_log_name=args.exp_name,
        validation_traces=val_traces,
    )


if __name__ == "__main__":
    t_start = time.time()
    main()
    print("time used: {:.2f}s".format(time.time() - t_start))
