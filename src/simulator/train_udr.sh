#!/bin/bash
set -e
# for range_id in 0 1 2 3 4 5 6 7 8 9; do
for range_id in 0 2 3 5 7 ; do
    save_dir=../../results/rand_5_dims_new/range_${range_id}
    # save_dir=../../results/tmp${range_id}
    mkdir -p ${save_dir}
    CUDA_VISIBLE_DEVICES="" python -m memory_profiler train.py --save-dir ${save_dir} \
        --config ../../config/train/rand_5_dims/rand${range_id}.json \
        --seed 43 > ${save_dir}/out.log &
    # CUDA_VISIBLE_DEVICES="" python test_rl.py --save-dir ${save_dir} \
    #     --seed 43
done
