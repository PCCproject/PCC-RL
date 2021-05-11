set -e

# for bw in 0 1 2 3 4 5 6; do
#     for config in 0 1 2 3 4 5 6 7 8 9 10; do
#         for i in 2; do # 2 3 4 5 6 7 8 9
#             echo ${bw}, ${config}, ${i}
#             python ../common/convert_mahimahi_format.py \
#                 --trace-dir /tank/zxxia/PCC-RL/results_0503/rand_bandwidth/${bw}/config_${config}/trace_${i} \
#                 --save-dir /tank/zxxia/PCC-RL/results_0503/rand_bandwidth/${bw}/config_${config}/trace_${i} &
#         done
#     done
# done

for delay in 5 50 100 150 200; do
    for config in 0 1 2 3 4 5 6 7 8 9 10; do
        for i in 2  ;do # 3 4 5 6 7 8 9
            # echo ${bw}, ${config}, ${i}
            python ../common/convert_mahimahi_format.py \
                --trace-dir /tank/zxxia/PCC-RL/results_0503/rand_delay/${delay}/config_${config}/trace_${i} \
                --save-dir /tank/zxxia/PCC-RL/results_0503/rand_delay/${delay}/config_${config}/trace_${i} &
        done
    done
done

# for T_s in 0 1 2 3 4 5 6; do
#     for config in 0 1 2 3 4 5 6 7 8 9 10; do
#         for i in 0 1 2;do # 3 4 5 6 7 8 9
#             # echo ${bw}, ${config}, ${i}
#             python ../common/convert_mahimahi_format.py \
#                 --trace-dir /tank/zxxia/PCC-RL/results_0503/rand_T_s/${T_s}/config_${config}/trace_${i} \
#                 --save-dir /tank/zxxia/PCC-RL/results_0503/rand_T_s/${T_s}/config_${config}/trace_${i}
#         done
#     done
# done
