import gym
import network_sim
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2


# multiprocess environment
n_cpu = 8
env = SubprocVecEnv([lambda: gym.make('PccNs-v0') for i in range(n_cpu)])

model = PPO2(MlpLstmPolicy, env, verbose=1, nminibatches=1, n_steps=1024, noptepochs=8)
model.learn(total_timesteps=(9600 * 410))
model.save("ppo2_custom_env")

export_dir = "/home/pcc/PCC/deep-learning/python/saved_models/stable_solve/"

with model.graph.as_default():

    pol = model.act_model

    obs_ph = pol.obs_ph
    state_ph = pol.states_ph
    act = pol.deterministic_action
    state = pol.snew
    sampled_act = pol.action
    mask_ph = pol.masks_ph

    obs_input = tf.saved_model.utils.build_tensor_info(obs_ph)
    state_input = tf.saved_model.utils.build_tensor_info(state_ph)
    mask_input = tf.saved_model.utils.build_tensor_info(mask_ph)
    outputs_tensor_info = tf.saved_model.utils.build_tensor_info(act)
    state_tensor_info = tf.saved_model.utils.build_tensor_info(state)
    stochastic_act_tensor_info = tf.saved_model.utils.build_tensor_info(sampled_act)
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"ob":obs_input, "state":state_input, "mask": mask_input},
        outputs={"act":outputs_tensor_info, "stochastic_act":stochastic_act_tensor_info, 
"state":state_tensor_info},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    #"""
    signature_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                     signature}

    model_builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    model_builder.add_meta_graph_and_variables(model.sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map=signature_map,
        clear_devices=True)
    model_builder.save(as_text=True)

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2_custom_env")

# Enjoy trained agent
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #env.render()
    print(obs)
    done = dones[0]
