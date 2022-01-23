
save_root=/datamirror/zxxia/PCC-RL/results_1006/compare_udr_rule_based


python compare_udr_rule_based.py \
    --heuristic bbr_old \
    --model-path $model_path \
    --config-file $config_file \
    --seed \
    --save-dir
