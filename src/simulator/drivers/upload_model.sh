# model_path=../../results_0423/udr_7_dims/bo_bw1/model_step_187200.ckpt
# model_path=../../results_0423/udr_7_dims/bo_bw1/model_step_352800.ckpt
# model_path=../../results_0423/udr_7_dims/bo_bw1/model_step_273600.ckpt
# model_path=../../results_0423/udr_7_dims/bo_d_bw0/model_step_273600.ckpt
# model_path=../../results_0423/udr_7_dims/bo_d_bw0/model_step_108000.ckpt
# model_path=../../results_0426/udr_7_dims/bo_bw0/model_step_525600.ckpt
# model_path=../../results_0426/udr_7_dims/bo_bw1/model_step_1684800.ckpt
# model_path=../../results_0426/udr_7_dims/range0/model_step_669600.ckpt
# model_path=../../results_0426/udr_7_dims/bo_delay0/model_step_1468800.ckpt

model_path=../../results_0430/udr_7_dims/bo_delay2/seed_10/model_step_576000.ckpt
model_path=../../results_0430/udr_7_dims/bo_delay3/seed_50/model_step_367200.ckpt
model_path=../../results_0430/udr_7_dims/bo_delay4/seed_50/model_step_122400.ckpt
remote=40.117.147.179
user=zxxia
parent_dir=$(basename $(dirname ${model_path}))
grand_parent_dir=$(basename $(dirname $(dirname ${model_path})))
echo ${parent_dir}
ssh ${user}@${remote} "mkdir -p /home/zxxia/pantheon/models/udr_7_dims/${grand_parent_dir}/${parent_dir}"
scp -r ${model_path}.* ${user}@${remote}:/home/zxxia/pantheon/models/udr_7_dims/${grand_parent_dir}/${parent_dir}
