import copy
import random
import numpy as np
import math

from common.utils import set_seed, read_json_file, write_json_file


udr_large = read_json_file(
    '../../config/train/udr_7_dims_0905/udr_large.json')[0]
    # '/tank/zxxia/PCC-RL/config/train/udr_7_dims_0826/udr_large.json')[0]


def gen_random_range(val_min, val_max, logscale=False, weight=1/3, single_point=False):
    if single_point:
        if logscale:
            new_val_min = random.uniform(math.log10(val_min), math.log10(val_max))
            new_val_max = new_val_min
            return 10**new_val_min, 10**new_val_max
        new_val_min = random.uniform(val_min, val_max)
        new_val_max = new_val_min
        return new_val_min, new_val_max

    if logscale:
        range_len = (math.log10(val_max) - math.log10(val_min)) * (1 - weight)
        # print(range_len)
        new_val_min = random.uniform(math.log10(
            val_min), math.log10(val_min) + range_len)
        new_val_max = new_val_min + \
            (np.log10(val_max) - np.log10(val_min)) * weight
        return 10**new_val_min, 10**new_val_max

    range_len = (val_max - val_min) * (1 - weight)
    # print(range_len)
    new_val_min = random.uniform(val_min, val_min + range_len)
    new_val_max = new_val_min + (val_max - val_min) * weight
    return new_val_min, new_val_max



for i in range(10, 110, 10):
    # print(i)
    set_seed(i)
    bw_upper_bnd_min, bw_upper_bnd_max = udr_large['bandwidth_upper_bound']
    new_bw_upper_bnd_min, new_bw_upper_bnd_max = gen_random_range(bw_upper_bnd_min, bw_upper_bnd_max, True)

    bw_lower_bnd_min, bw_lower_bnd_max = udr_large['bandwidth_lower_bound']
    new_bw_lower_bnd_min, new_bw_lower_bnd_max = gen_random_range(bw_lower_bnd_min, new_bw_upper_bnd_max, True)
    while new_bw_lower_bnd_min > new_bw_upper_bnd_min:
        # print('while', new_bw_lower_bnd_min, new_bw_upper_bnd_min)
        new_bw_lower_bnd_min, new_bw_lower_bnd_max = gen_random_range(bw_lower_bnd_min, new_bw_upper_bnd_max, True)

    delay_min, delay_max = udr_large['delay']
    new_delay_min, new_delay_max = gen_random_range(delay_min, delay_max)
    loss_min, loss_max = udr_large['loss']
    new_loss_min, new_loss_max = gen_random_range(loss_min, loss_max)
    queue_min, queue_max = udr_large['queue']
    new_queue_min, new_queue_max = gen_random_range(queue_min, queue_max)
    T_s_min, T_s_max = udr_large['T_s']
    new_T_s_min, new_T_s_max = gen_random_range(T_s_min, T_s_max)
    delay_noise_min, delay_noise_max = udr_large['delay_noise']
    new_delay_noise_min, new_delay_noise_max = gen_random_range(delay_noise_min, delay_noise_max)

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

    write_json_file('../../config/train/udr_7_dims_0905/udr_mid_seed_{}.json'.format(i), [udr_mid])


    set_seed(i)
    bw_upper_bnd_min, bw_upper_bnd_max = udr_large['bandwidth_upper_bound']
    new_bw_upper_bnd_min, new_bw_upper_bnd_max = gen_random_range(bw_upper_bnd_min, bw_upper_bnd_max, True, single_point=True)
    new_bw_lower_bnd_min, new_bw_lower_bnd_max = new_bw_upper_bnd_min, new_bw_upper_bnd_max

    delay_min, delay_max = udr_large['delay']
    new_delay_min, new_delay_max = gen_random_range(delay_min, delay_max, False, 1/9, single_point=True)
    loss_min, loss_max = udr_large['loss']
    new_loss_min, new_loss_max = gen_random_range(loss_min, loss_max, False, 1/9, single_point=True)
    queue_min, queue_max = udr_large['queue']
    new_queue_min, new_queue_max = gen_random_range(queue_min, queue_max, False, 1/9, single_point=True)
    T_s_min, T_s_max = udr_large['T_s']
    new_T_s_min, new_T_s_max = gen_random_range(T_s_min, T_s_max, False, 1/9, single_point=True)
    delay_noise_min, delay_noise_max = udr_large['delay_noise']
    new_delay_noise_min, new_delay_noise_max = gen_random_range(delay_noise_min, delay_noise_max, False, 1/9, single_point=True)

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

    write_json_file('../../config/train/udr_7_dims_0905/udr_small_seed_{}.json'.format(i), [udr_small])




    # bw_upper_bnd_min, bw_upper_bnd_max = udr_large['bandwidth_upper_bound']
    # new_bw_upper_bnd_min, new_bw_upper_bnd_max = gen_random_range(bw_upper_bnd_min, bw_upper_bnd_max, True, 1/9)
    # bw_lower_bnd_min, bw_lower_bnd_max = udr_large['bandwidth_lower_bound']
    # new_bw_lower_bnd_min, new_bw_lower_bnd_max = gen_random_range(bw_lower_bnd_min, new_bw_upper_bnd_max, True, 1/9)
    # while new_bw_lower_bnd_min > new_bw_upper_bnd_min:
        # new_bw_lower_bnd_min, new_bw_lower_bnd_max = gen_random_range(bw_lower_bnd_min, new_bw_upper_bnd_max, True, 1/9)
#
    # delay_min, delay_max = udr_large['delay']
    # new_delay_min, new_delay_max = gen_random_range(delay_min, delay_max, False, 1/9)
    # loss_min, loss_max = udr_large['loss']
    # new_loss_min, new_loss_max = gen_random_range(loss_min, loss_max, False, 1/9)
    # queue_min, queue_max = udr_large['queue']
    # new_queue_min, new_queue_max = gen_random_range(queue_min, queue_max, False, 1/9)
    # T_s_min, T_s_max = udr_large['T_s']
    # new_T_s_min, new_T_s_max = gen_random_range(T_s_min, T_s_max, False, 1/9)
#
    # udr_small = copy.deepcopy(udr_large)
    # udr_small['bandwidth_lower_bound'][0] = new_bw_lower_bnd_min
    # udr_small['bandwidth_lower_bound'][1] = new_bw_lower_bnd_max
    # udr_small['bandwidth_upper_bound'][0] = new_bw_upper_bnd_min
    # udr_small['bandwidth_upper_bound'][1] = new_bw_upper_bnd_max
    # udr_small['delay'][0] = new_delay_min
    # udr_small['delay'][1] = new_delay_max
    # udr_small['loss'][0] = new_loss_min
    # udr_small['loss'][1] = new_loss_max
    # udr_small['queue'][0] = int(new_queue_min)
    # udr_small['queue'][1] = int(new_queue_max)
    # udr_small['T_s'][0] = new_T_s_min
    # udr_small['T_s'][1] = new_T_s_max
#
    # write_json_file('../../config/train/udr_7_dims_0904/udr_small_seed_{}.json'.format(i), [udr_small])
