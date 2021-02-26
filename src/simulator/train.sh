#!/bin/bash
set -e

# mkdir -p ../../results/rand_bw/bw_50_100
# CUDA_VISIBLE_DEVICES="" python train.py --save-dir ../../results/rand_bw/bw_50_100 \
#     --seed 43 --range-id 0 > ../../results/rand_bw/bw_50_100/out.log &
#
# mkdir -p ../../results/rand_bw/bw_50_500
# CUDA_VISIBLE_DEVICES="" python train.py --save-dir ../../results/rand_bw/bw_50_500 \
#     --seed 43 --range-id 1 > ../../results/rand_bw/bw_50_500/out.log &
#
# mkdir -p ../../results/rand_bw/bw_50_1000
# CUDA_VISIBLE_DEVICES="" python train.py --save-dir ../../results/rand_bw/bw_50_1000 \
#     --seed 43 --range-id 2 > ../../results/rand_bw/bw_50_1000/out.log &
#
# mkdir -p ../../results/rand_bw/bw_50_1500
# CUDA_VISIBLE_DEVICES="" python train.py --save-dir ../../results/rand_bw/bw_50_1500 \
#     --seed 43 --range-id 3 > ../../results/rand_bw/bw_50_1500/out.log &
#
# mkdir -p ../../results/rand_bw/bw_50_2000
# CUDA_VISIBLE_DEVICES="" python train.py --save-dir ../../results/rand_bw/bw_50_2000 \
#     --seed 43 --range-id 4 > ../../results/rand_bw/bw_50_2000/out.log &
#
# mkdir -p ../../results/rand_bw/bw_50_3000
# CUDA_VISIBLE_DEVICES="" python train.py --save-dir ../../results/rand_bw/bw_50_3000 \
#     --seed 43 --range-id 5 > ../../results/rand_bw/bw_50_3000/out.log &
#
# mkdir -p ../../results/rand_bw/bw_50_5000
# CUDA_VISIBLE_DEVICES="" python train.py --save-dir ../../results/rand_bw/bw_50_5000 \
#     --seed 43 --range-id 6 > ../../results/rand_bw/bw_50_5000/out.log &

# for range_id in 0 1 2 3 4 5 6 7 8; do
for range_id in 100 500 1000 1500 2000 3000 5000; do
# save_dir=../../results/rand_queue/queue_0_${range_id}
save_dir=../../results/rand_bw/bw_50_${range_id}
mkdir -p ${save_dir}
# CUDA_VISIBLE_DEVICES="" python train.py --save-dir ${save_dir} \
#     --config ../../config/train/rand_queue/rand_queue${range_id}.json \
#     --seed 43 > ${save_dir}/out.log &
CUDA_VISIBLE_DEVICES="" python test_rl.py --save-dir ${save_dir} \
    --seed 43
done
