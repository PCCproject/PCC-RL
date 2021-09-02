from genericpath import exists
import os

import numpy as np
from common.utils import read_json_file, set_seed

from simulator.trace import Trace, generate_bw_delay_series

# from simulator.trace.generate_trace, generate_bw_delay_series

n_traces = 10
save_dir = '../../data/synthetic_traces_30s'

set_seed = 20
# ret_trace = Trace([30], [1.0], [70], 0, 10)
# os.makedirs(os.path.join(save_dir, "rand_bandwidth", str(round(bw, 2))), exist_ok=True)
# ret_trace.dump('trace.json')

DELAY = 50
BW = 1
QUEUE = 10
DURATION = 30
LOSS = 0

# for bw in np.linspace(np.log10(0.6), np.log10(6), 10):
#     DELAY = 50
#     QUEUE = 10
#     bandwidth_range = [10**bw, 10**bw]
#     bw = round(10**bw, 2)
#     delay_range = [DELAY, DELAY]
#     d_bw = 0
#     d_delay = 0
#     T_s = 1
#     duration = DURATION
#     loss_rate = LOSS
#     queue_size = QUEUE
#     for i in range(n_traces):
#         timestamps, bandwidths, delays = generate_bw_delay_series(
#             T_s, duration, bandwidth_range[0], bandwidth_range[1],
#             delay_range[0], delay_range[1])
#         ret_trace = Trace(timestamps, bandwidths,
#                           delays, loss_rate, queue_size)
#         os.makedirs(os.path.join(save_dir, "rand_bandwidth", str(round(bw, 2))), exist_ok=True)
#         ret_trace.dump(os.path.join(save_dir, "rand_bandwidth",
#                                     str(round(bw, 2)), 'trace{:04d}.json'.format(i)))


for delay in range(10, 110, 10):
    # bandwidth_range = [0.1, 2]
    bandwidth_range = [1, 6]
    delay_range = [delay, delay]
    d_bw = 0
    d_delay = 0
    T_s = 4
    duration = DURATION
    loss_rate = LOSS
    queue_size = QUEUE
    for i in range(n_traces):
        timestamps, bandwidths, delays = generate_bw_delay_series(
            T_s, duration, bandwidth_range[0], bandwidth_range[1],
            delay_range[0], delay_range[1])
        ret_trace = Trace(timestamps, bandwidths,
                          delays, loss_rate, queue_size)
        os.makedirs(os.path.join(save_dir, "rand_delay", str(delay)), exist_ok=True)
        ret_trace.dump(os.path.join(save_dir, "rand_delay",
                                    str(round(delay, 2)), 'trace{:04d}.json'.format(i)))
#
#
# for loss in np.arange(0, 0.11, 0.01):
#     bandwidth_range = [BW, BW]
#     delay_range = [DELAY, DELAY]
#     d_bw = 0
#     d_delay = 0
#     T_s = 0
#     duration = DURATION
#     loss_rate = loss
#     queue_size = QUEUE
#     for i in range(n_traces):
#         timestamps, bandwidths, delays = generate_bw_delay_series(
#             d_bw, d_delay, T_s, duration, bandwidth_range[0], bandwidth_range[1],
#             delay_range[0], delay_range[1])
#         ret_trace = Trace(timestamps, bandwidths,
#                           delays, loss_rate, queue_size)
#         os.makedirs(os.path.join(save_dir, "rand_loss", str(loss)), exist_ok=True)
#         ret_trace.dump(os.path.join(save_dir, "rand_loss",
#                                     str(round(loss, 2)), 'trace{:04d}.json'.format(i)))
#
#
#
# for queue in list(range(2, 10, 2)) + list(range(10, 210, 10)):
#     bandwidth_range = [BW, BW]
#     delay_range = [DELAY, DELAY]
#     d_bw = 0
#     d_delay = 0
#     T_s = 0
#     duration = DURATION
#     loss_rate = LOSS
#     queue_size = queue
#     for i in range(n_traces):
#         timestamps, bandwidths, delays = generate_bw_delay_series(
#             d_bw, d_delay, T_s, duration, bandwidth_range[0], bandwidth_range[1],
#             delay_range[0], delay_range[1])
#         ret_trace = Trace(timestamps, bandwidths,
#                           delays, loss_rate, queue_size)
#         os.makedirs(os.path.join(save_dir, "rand_queue", str(queue)), exist_ok=True)
#         ret_trace.dump(os.path.join(save_dir, "rand_queue",
#                                     str(round(queue, 2)), 'trace{:04d}.json'.format(i)))
#
#
# for duration in list(range(10, 32, 2)):
#     bandwidth_range = [BW, BW]
#     delay_range = [DELAY, DELAY]
#     d_bw = 0
#     d_delay = 0
#     T_s = 1
#     duration = duration
#     loss_rate = LOSS
#     queue_size = QUEUE
#     for i in range(n_traces):
#         timestamps, bandwidths, delays = generate_bw_delay_series(
#             d_bw, d_delay, T_s, duration, bandwidth_range[0], bandwidth_range[1],
#             delay_range[0], delay_range[1])
#         ret_trace = Trace(timestamps, bandwidths,
#                           delays, loss_rate, queue_size)
#         os.makedirs(os.path.join(save_dir, "rand_duration", str(duration)), exist_ok=True)
#         ret_trace.dump(os.path.join(save_dir, "rand_duration",
#                                     str(round(duration, 2)), 'trace{:04d}.json'.format(i)))
#

# for t_s in [1, 2, 5, 10]: #np.arange(0, 7, 0.5):
#     bandwidth_range = [1, 6]
#     delay_range = [DELAY, DELAY]
#     d_bw = 0.2
#     d_delay = 0
#     T_s = t_s
#     duration = 10
#     loss_rate = LOSS
#     queue_size = QUEUE
#     for i in range(n_traces):
#         timestamps, bandwidths, delays = generate_bw_delay_series(
#             T_s, duration, bandwidth_range[0], bandwidth_range[1],
#             delay_range[0], delay_range[1])
#         ret_trace = Trace(timestamps, bandwidths,
#                           delays, loss_rate, queue_size)
#         os.makedirs(os.path.join(save_dir, "rand_T_s", str(round(T_s, 2))), exist_ok=True)
#         ret_trace.dump(os.path.join(save_dir, "rand_T_s",
#                                     str(round(T_s, 2)), 'trace{:04d}.json'.format(i)))

# for d_bw in np.arange(0, 6, 1):
#     bandwidth_range = [1, 1+d_bw]
#     delay_range = [DELAY, DELAY]
#     d_bw = round(d_bw, 2)
#     d_delay = 0
#     T_s = 1
#     duration = 30
#     loss_rate = LOSS
#     queue_size = QUEUE
#     for i in range(n_traces):
#         timestamps, bandwidths, delays = generate_bw_delay_series(
#             T_s, duration, bandwidth_range[0], bandwidth_range[1],
#             delay_range[0], delay_range[1])
#         ret_trace = Trace(timestamps, bandwidths,
#                           delays, loss_rate, queue_size)
#         os.makedirs(os.path.join(save_dir, "rand_d_bw", str(round(d_bw, 2))), exist_ok=True)
#         ret_trace.dump(os.path.join(save_dir, "rand_d_bw",
#                                     str(round(d_bw, 2)), 'trace{:04d}.json'.format(i)))
#
#
# # for d_delay in np.arange(0, 0.6, 0.1):
# #     bandwidth_range = [BW, BW]
# #     delay_range = [5, 100]
# #     d_bw = 0
# #     d_delay = round(d_delay, 2)
# #     T_s = 1
# #     duration = 30
# #     loss_rate = 0
# #     queue_size = 2
# #     for i in range(n_traces):
# #         timestamps, bandwidths, delays = generate_bw_delay_series(
# #             d_bw, d_delay, T_s, duration, bandwidth_range[0], bandwidth_range[1],
# #             delay_range[0], delay_range[1])
# #         ret_trace = Trace(timestamps, bandwidths,
# #                           delays, loss_rate, queue_size)
# #         os.makedirs(os.path.join(save_dir, "rand_d_delay", str(d_delay)), exist_ok=True)
# #         ret_trace.dump(os.path.join(save_dir, "rand_d_delay",
# #                                     str(round(d_delay, 2)), 'trace{:04d}.json'.format(i)))


# for delay_noise in np.arange(0, 1, 0.1)*100:
#     bandwidth_range = [BW, BW]
#     delay_range = [DELAY, DELAY]
#     d_bw = 0
#     d_delay = 0
#     delay_noise = round(delay_noise, 2)
#     T_s = 0
#     duration = 30
#     loss_rate = 0
#     queue_size = QUEUE
#     for i in range(n_traces):
#         timestamps, bandwidths, delays = generate_bw_delay_series(
#             T_s, duration, bandwidth_range[0], bandwidth_range[1],
#             delay_range[0], delay_range[1])
#         ret_trace = Trace(timestamps, bandwidths, delays, loss_rate, queue_size,
#                           delay_noise)
#         os.makedirs(os.path.join(save_dir, "rand_delay_noise", str(delay_noise)), exist_ok=True)
#         ret_trace.dump(os.path.join(save_dir, "rand_delay_noise",
#                                     str(round(delay_noise, 2)), 'trace{:04d}.json'.format(i)))
