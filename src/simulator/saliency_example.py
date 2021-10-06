"""
To run this script, modify MODEL_PATH and SAVE_DIR at line 15-16.

Then, run cmd:
CUDA_VISIBLE_DEVICES="" python saliency_example.py
"""
import os
import time

from simulator.aurora import Aurora, test_on_traces
from simulator.network_simulator.bbr import BBR
from simulator.network_simulator.bbr_old import BBR_old
from simulator.network_simulator.cubic import Cubic
from simulator.trace import generate_trace, Trace
from common.utils import set_seed


MODEL_PATH = "../../results_0826/udr_6/udr_start/seed_20/model_step_151200.ckpt"
SAVE_DIR = 'test_saliency'

def main():
    set_seed(42)
    dummy_trace = generate_trace(
        duration_range=(10, 10),
        bandwidth_lower_bound_range=(0.1, 0.1),
        bandwidth_upper_bound_range=(12, 12),
        delay_range=(25, 25), loss_rate_range=(0.0, 0.0), queue_size_range=(1, 1),
        T_s_range=(3, 3), delay_noise_range=(0, 0))
    dummy_trace.dump(os.path.join(SAVE_DIR, "test_trace.json"))

    genet = Aurora(seed=20, log_dir=SAVE_DIR, pretrained_model_path=MODEL_PATH,
                   timesteps_per_actorbatch=10, record_pkt_log=True)
    t_start = time.time()
    print(genet.test(dummy_trace, SAVE_DIR, True, saliency=True))
    print("aurora", time.time() - t_start)

    # bbr = BBR(True)
    # t_start = time.time()
    # print(bbr.test(dummy_trace, SAVE_DIR, True))
    # print('bbr', time.time() - t_start)
    #
    # bbr_old = BBR_old(True)
    # t_start = time.time()
    # print(bbr_old.test(dummy_trace, SAVE_DIR, True))
    # print('bbr_old', time.time() - t_start)

    # cubic = Cubic(False)
    # t_start = time.time()
    # print(cubic.test(dummy_trace, SAVE_DIR, False))
    # print('cubic',time.time() - t_start)

if __name__ == '__main__':
    main()
