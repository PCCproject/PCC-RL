import argparse
import copy
import os
import random
import math

import numpy as np

from common.utils import set_seed, read_json_file, write_json_file


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Generate different udr ranges.")
    parser.add_argument('--config-file', type=str, required=True,
                        help="path to a udr large config file.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="path to save.")

    args, unknown = parser.parse_known_args()
    return args


def gen_random_range(dim: str, val_min: float, val_max: float,
                     logscale: bool = False, weight: float =1/3,
                     single_point: bool = False):
    if single_point:
        if logscale:
            if dim == 'loss':
                exponent = float(np.random.uniform(np.log10(val_min+1e-5), np.log10(val_max+1e-5), 1))
                if exponent < -4:
                    new_val = 0
                else:
                    new_val = 10**exponent
                return new_val, new_val
            new_val_min = random.uniform(
                math.log10(val_min), math.log10(val_max))
            new_val_max = new_val_min
            return 10**new_val_min, 10**new_val_max
        new_val_min = random.uniform(val_min, val_max)
        new_val_max = new_val_min
        return new_val_min, new_val_max

    if logscale:
        if dim == 'loss':
            range_len = (math.log10(val_max+1e-5) - math.log10(val_min+1e-5)) * (1 - weight)
            new_val_min = random.uniform(math.log10(
                val_min+1e-5), math.log10(val_min+1e-5) + range_len)
            new_val_max = new_val_min + \
                (np.log10(val_max+1e-5) - np.log10(val_min+1e-5)) * weight
            if new_val_min < -4:
                new_val_min = 0
            else:
                new_val_min = 10**new_val_min
            if new_val_max < -4:
                new_val_max = 0
            else:
                new_val_max = 10**new_val_max
            return new_val_min, new_val_max
        range_len = (math.log10(val_max) - math.log10(val_min)) * (1 - weight)
        new_val_min = random.uniform(math.log10(
            val_min), math.log10(val_min) + range_len)
        new_val_max = new_val_min + \
            (np.log10(val_max) - np.log10(val_min)) * weight
        return 10**new_val_min, 10**new_val_max

    range_len = (val_max - val_min) * (1 - weight)
    new_val_min = random.uniform(val_min, val_min + range_len)
    new_val_max = new_val_min + (val_max - val_min) * weight
    return new_val_min, new_val_max


def main():
    args = parse_args()
    udr_large = read_json_file(args.config_file)[0]

    for i in range(10, 110, 10):
        set_seed(i)
        bw_upper_bnd_min, bw_upper_bnd_max = udr_large['bandwidth_upper_bound']
        new_bw_upper_bnd_min, new_bw_upper_bnd_max = gen_random_range(
            'bandwidth_upper_bound', bw_upper_bnd_min, bw_upper_bnd_max, True)

        bw_lower_bnd_min, bw_lower_bnd_max = udr_large['bandwidth_lower_bound']
        new_bw_lower_bnd_min, new_bw_lower_bnd_max = gen_random_range(
            'bandwidth_lower_bound', bw_lower_bnd_min, new_bw_upper_bnd_max, True)
        while new_bw_lower_bnd_min > new_bw_upper_bnd_min:
            new_bw_lower_bnd_min, new_bw_lower_bnd_max = gen_random_range(
                'bandwidth_lower_bound', bw_lower_bnd_min, new_bw_upper_bnd_max, True)

        delay_min, delay_max = udr_large['delay']
        new_delay_min, new_delay_max = gen_random_range('delay', delay_min, delay_max)
        loss_min, loss_max = udr_large['loss']
        new_loss_min, new_loss_max = gen_random_range('loss', loss_min, loss_max, True)
        queue_min, queue_max = udr_large['queue']
        new_queue_min, new_queue_max = gen_random_range('queue', queue_min, queue_max)
        T_s_min, T_s_max = udr_large['T_s']
        new_T_s_min, new_T_s_max = gen_random_range('T_s', T_s_min, T_s_max)
        delay_noise_min, delay_noise_max = udr_large['delay_noise']
        new_delay_noise_min, new_delay_noise_max = gen_random_range(
            'delay_noise', delay_noise_min, delay_noise_max)

        udr_mid = copy.deepcopy(udr_large)
        udr_mid['bandwidth_lower_bound'][0] = new_bw_lower_bnd_min
        udr_mid['bandwidth_lower_bound'][1] = new_bw_lower_bnd_max
        udr_mid['bandwidth_upper_bound'][0] = new_bw_upper_bnd_min
        udr_mid['bandwidth_upper_bound'][1] = new_bw_upper_bnd_max
        udr_mid['delay'][0] = new_delay_min
        udr_mid['delay'][1] = new_delay_max
        udr_mid['loss'][0] = new_loss_min
        udr_mid['loss'][1] = new_loss_max
        udr_mid['queue'][0] = new_queue_min
        udr_mid['queue'][1] = new_queue_max
        udr_mid['T_s'][0] = new_T_s_min
        udr_mid['T_s'][1] = new_T_s_max
        udr_mid['delay_noise'][0] = new_delay_noise_min
        udr_mid['delay_noise'][1] = new_delay_noise_max

        write_json_file(os.path.join(args.save_dir, 'udr_mid_seed_{}.json'.format(i)), [udr_mid])

        set_seed(i)
        bw_upper_bnd_min, bw_upper_bnd_max = udr_large['bandwidth_upper_bound']
        new_bw_upper_bnd_min, new_bw_upper_bnd_max = gen_random_range(
            'bandwidth_upper_bound', bw_upper_bnd_min, bw_upper_bnd_max, True, single_point=True)
        new_bw_lower_bnd_min, new_bw_lower_bnd_max = new_bw_upper_bnd_min, new_bw_upper_bnd_max

        delay_min, delay_max = udr_large['delay']
        new_delay_min, new_delay_max = gen_random_range(
            'delay', delay_min, delay_max, False, 1/9, single_point=True)
        loss_min, loss_max = udr_large['loss']
        new_loss_min, new_loss_max = gen_random_range(
            'loss', loss_min, loss_max, True, 1/9, single_point=True)
        queue_min, queue_max = udr_large['queue']
        new_queue_min, new_queue_max = gen_random_range(
            'queue', queue_min, queue_max, False, 1/9, single_point=True)
        T_s_min, T_s_max = udr_large['T_s']
        new_T_s_min, new_T_s_max = gen_random_range(
            'T_s', T_s_min, T_s_max, False, 1/9, single_point=True)
        delay_noise_min, delay_noise_max = udr_large['delay_noise']
        new_delay_noise_min, new_delay_noise_max = gen_random_range(
            'delay_noise', delay_noise_min, delay_noise_max, False, 1/9, single_point=True)

        udr_small = copy.deepcopy(udr_large)
        udr_small['bandwidth_lower_bound'][0] = new_bw_lower_bnd_min
        udr_small['bandwidth_lower_bound'][1] = new_bw_lower_bnd_max
        udr_small['bandwidth_upper_bound'][0] = new_bw_upper_bnd_min
        udr_small['bandwidth_upper_bound'][1] = new_bw_upper_bnd_max
        udr_small['delay'][0] = new_delay_min
        udr_small['delay'][1] = new_delay_max
        udr_small['loss'][0] = new_loss_min
        udr_small['loss'][1] = new_loss_max
        udr_small['queue'][0] = new_queue_min
        udr_small['queue'][1] = new_queue_max
        udr_small['T_s'][0] = new_T_s_min
        udr_small['T_s'][1] = new_T_s_max
        udr_small['delay_noise'][0] = new_delay_noise_min
        udr_small['delay_noise'][1] = new_delay_noise_max

        write_json_file(os.path.join(args.save_dir, 'udr_small_seed_{}.json'.format(i)), [udr_small])

if __name__ == "__main__":
    main()
