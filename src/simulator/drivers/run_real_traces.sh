set -e
root=/tank/zxxia/PCC-RL/data
# save_root=../../results_0515/test_real_traces
# save_root=../../results_0515/test_real_traces_heuristic_stateless_fix_max_tput_heuristic_thresh_0.05
save_root=../../results_0515/test_real_traces_heuristic_fix
mkdir -p ${save_root}

# model_path=../../results_0426/udr_7_dims/bo_delay0/model_step_1468800.ckpt
# model_path=../../results_0430/udr_7_dims/large_bw_long_rtt/model_step_806400.ckpt
# model_path=../../results_0426/udr_7_dims/range2/model_step_1245600.ckpt

# model_path=/tank/zxxia/PCC-RL/results_0430/udr_7_dims/large_bw_long_rtt/seed_50/model_step_158400.ckpt
# model_path=/tank/zxxia/PCC-RL/results_0430/udr_7_dims/large_bw_long_rtt/seed_50/model_step_453600.ckpt
# model_path=/tank/zxxia/PCC-RL/results_0430/udr_7_dims/range2/seed_50/model_step_1137600.ckpt
model_path=../../results_0430/udr_7_dims/bo_delay2/seed_10/model_step_576000.ckpt
model_path=../../results_0430/udr_7_dims/bo_delay2/seed_50/model_step_1562400.ckpt
model_path=../../results_0430/udr_7_dims/bo_delay3/seed_50/model_step_367200.ckpt
model_path=../../results_0430/udr_7_dims/bo_delay4/seed_50/model_step_122400.ckpt
model_path=../../results_0430/udr_7_dims/bo_delay5/seed_50/model_step_93600.ckpt
model_path=../../results_0503/udr_7_dims/udr_large/seed_50/model_step_396000.ckpt
model_path="../../results_0515/udr_7_dims_fix_start/udr_large_lossless/seed_20/model_step_1317600.ckpt"
model_path="../../results_0515/udr_mid_simple_stateless/udr_mid_simple/seed_20/model_step_633600.ckpt"
# model_path=../../results_0426/udr_7_dims/bo_bw1/model_step_1684800.ckpt
# model_path=../../results_0415/udr_7_dims/range1/model_step_4262400.ckpt
# model_path=../../results_0415/udr_7_dims/range0_vary_bw_cont/model_step_1850400.ckpt
printf "flow,aurora_tput,aurora_lat,aurora_loss,aurora_reward,cubic_tput,cubic_lat,cubic_loss,cubic_reward,bw_avg,bw_std,t_s_bw,change_freq,bw_range,rtt_avg\n" > ${save_root}/summary.csv
for scene in cellular; do # wireless ; do
    links=$(ls -d ${root}/${scene}/*/)
    # echo ${links}
    for link in ${links}; do
        link_name=$(basename $link)

        # if [[ ${link_name} == *"36"* ]] || [[ ${link_name} == *"India"* ]]; then
        #     continue
        # fi
        # if [[ ${link_name} != *"China"* ]]; then
        #     continue
        # fi
        echo ${link}
        traces=$(ls ${link}*_datalink_run*.log)
        for trace in ${traces}; do
            echo ${trace}
            if [[ ${trace} != *"bbr"* ]] && [[ ${trace} != *"vegas"* ]] ; then
                continue
            fi
# && [[ ${trace} != *"cubic"* ]]

            run_name=$(basename ${trace} .log)
            python evaluate_cubic.py \
                --trace-file ${trace} --loss 0 --queue 10 \
                --save-dir ${save_root}/${scene}/${link_name}/${run_name}
            CUDA_VISIBLE_DEVICES="" python evaluate_aurora.py \
                --trace-file ${trace} \
                --loss 0 \
                --queue 10 \
                --model-path ${model_path} \
                --save-dir ${save_root}/${scene}/${link_name}/${run_name}
            # # sleep 1
            python ../plot_scripts/plot_time_series.py \
                --trace-file ${trace} \
                --log-file ${save_root}/${scene}/${link_name}/${run_name}/aurora_simulation_log.csv \
                ${save_root}/${scene}/${link_name}/${run_name}/cubic_simulation_log.csv \
                --save-dir ${save_root}/${scene}/${link_name}/${run_name}/

            python ../plot_scripts/plot_packet_log.py \
                --log-file ${save_root}/${scene}/${link_name}/${run_name}/aurora_packet_log.csv \
                ${save_root}/${scene}/${link_name}/${run_name}/cubic_packet_log.csv \
                --save-dir ${save_root}/${scene}/${link_name}/${run_name}/  \
                --trace-file ${trace} >> ${save_root}/summary.csv
            printf "\n" >> ${save_root}/summary.csv
            # break

        done
        # break
    done
done



