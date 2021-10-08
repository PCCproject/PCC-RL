import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Plot saliency map.")
    parser.add_argument('--saliency-log', type=str,  required=True,
                        help="path to a .npy saliency log file.")
    parser.add_argument('--save-dir', type=str, default=None,
                        help="path to save the plot.")

    args, unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20

    with open(args.saliency_log, 'rb') as f:
        grad = np.load(f)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        grad_input = grad.T
        im = ax.imshow(np.abs(grad_input))
        ax.set_xlabel('step')
        ax.set_ylabel('feature window')
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, orientation='horizontal')
        cbar.ax.set_xlabel("saliency")
        # print(np.mean(np.abs(grad_input), axis=1))

        fig.tight_layout()

        # im.set_clim(vmax=0, vmin=-200)
        fig.savefig(os.path.join(args.save_dir, 'saliency_map.jpg'), bbox_inches='tight')

if __name__ == '__main__':
    main()
