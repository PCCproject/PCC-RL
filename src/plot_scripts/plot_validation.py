import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


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
    for log_file in args.log_file:
        plt.figure()
        model_name = log_file.split('/')[-2]
        plt.title(model_name)
        df = pd.read_csv(log_file, sep='\t')
        best_step = int(df['n_calls'][df['mean_validation_reward'].argmax()])
        t_used = df['tot_t_used(min)'][df['mean_validation_reward'].argmax()]
        best_reward = df['mean_validation_reward'].max()
        best_model_path = os.path.join(os.path.dirname(log_file), "model_step_{}.ckpt.meta".format(best_step))
        if not os.path.exists(best_model_path):
            best_step += 252000
            best_model_path = os.path.join(os.path.dirname(log_file), "model_step_{}.ckpt.meta".format(best_step))

        plt.plot(df['n_calls'], df['mean_validation_reward'],
                 'o-', label="best_reward: {:.2f}, best step: {}, used {:.2f}min".format(best_reward, int(best_step), t_used))
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
