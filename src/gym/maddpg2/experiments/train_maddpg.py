'''
RUN:
    # python3 train_maddpg.py --max-episode-len 100 --num-agents 1 --log-dir 1e1 --num-episodes 40000 --lr 1e-3
    # python3 train_maddpg.py --max-episode-len 100 --num-agents 2 --log-dir 2e1 --num-episodes 40000 --lr 1e-3
RUN DOUBLE AGENT WITH DEGUBBER:
    # python3 train_maddpg.py --max-episode-len 100 --save-rate 100 --num-agents 2 --log-dir 210 --num-episodes 40000 --lr 1e-3 --debug
    # WHEN SEE DEBUGER WINDOW, ENTER: run -f has_inf_or_nan
    # WAIT FOR NAN TO APPEAR THAN TRACE BACK THE PROBLEMATIC LAYERS
TO MAKE GRAPHS FROM DUMP-EVENTS1 (ON SINGLE AGENT):
    # python3 make_graph.py --num-agents 2 --dump-rate 100 --save-rate 100 --in-out 2e19 2e19 --log-range 1 190 --type epi
TO RUN MULTI-PARTICLE ENVIRONMENT, SEE GITHUB PAGE, SEARCH OPENAI-MADDPG:
    # python3 train_maddpg.py --scenario simple_spread


    # source ~/.bashrc
    # python3 train_maddpg.py --max-episode-len 100 --save-rate 50 --num-agents 2 --log-dir 210 --num-episodes 30000 --lr 1e-3 --debug
    # python3 train_maddpg.py --max-episode-len 100 --save-rate 100 --num-agents 1 --log-dir 31 --num-episodes 30000 --lr 1e-3

    -- experiment log --
    baseline: model3, no restriction on out range, works for 1 agent but blows up for multi-agents
    improve: model4, tanh to change to (-1, 1), then scale it by multiplying, if unscaled learns slow or does not learn
    1e1e1: agent: 1, multiply-layer: 10, config.DELTA_SCALE = 0.025, p_reg * 1e-3
    1e6: ran on simple network simulator

    2e1e1: agent: 2, multiply-layer: 10, config.DELTA_SCALE = 0.025, p_reg * 1e-3
    2e1e2: agent: 2, multiply-layer: 9, config.DELTA_SCALE = 0.025, p_reg * 1e-3
    2e1e3: agent: 2, multiply-layer: 7, config.DELTA_SCALE = 0.025, p_reg * 1e-1, avg_rew = 40 not fair, send = 43, 48
    2e1e4: agent: 2, multiply-layer: 7, config.DELTA_SCALE = 0.025, p_reg * 1e0, avg_rew = 40, send = 45
    2e5: agent: 2, multiply-layer: 10, config.DELTA_SCALE = 0.025, p_reg * 1e-2
    2e6: agent: 2, multiply-layer: 9, config.DELTA_SCALE = 0.025, p_reg * 1e-3, penalty = 0.25
    2e7: agent: 2, multiply-layer: 9, config.DELTA_SCALE = 0.025, p_reg * 1e-3, penalty = 1.0
    2e8: original model, relu std, config.DELTA_SCALE = 0.025, p_reg * 1e-3
    2e9: original model, elu std, config.DELTA_SCALE = 0.025, p_reg * 1e-3
    2e10: repeat 2e6
    2e11: repeat 2e1e2
    2e12: repeat 2e1e2, but with latency reward turned off, only throughput and lossrate
    2e13: repeat 2e1e2, throughput only
    2e14: 2018 Pcc, sendrate - 11.3 * sendrate * lossrate
    2e15: 2015 Pcc, throughput * Sigmoid(lossrate−0.05) − sendrate * lossrate
    2e16: ran on simple network simulator, no P term, 2 obs
    2e17: simple, P term, 4 obs
    2e18: P term, sending_time noise, high bw, high min_rate, small queue_size, long run_dur

        turn off  F
        variance 1%-9%
        bw = 400, 80, longer takes bigger T
        qsize = 10-50
        run_dur = 1.0-2.0

        (base)bw=400, min_r=80, queue = 10, no F, variance = 0.1, run_dur = 1.0
        2e22: queue = 10 (base), no F
        2e23: queue = 50, no F
        2e24: queue = 10, F
        2e25: queue = 10, no F, 800, 160
        2e26:  queue = 10 (base), no F, variance = 0.25
        2e27: run_dur = 2.0

    3e1e1: agent: 3, multiply-layer: 8.9, config.DELTA_SCALE = 0.025, p_reg * 1e-3
    3e1e2: agent: 3, multiply-layer: 7, config.DELTA_SCALE = 0.025, p_reg * 1e-3
    3e3: ran on simple network simulator
    3e4: simple sim, P term, 2 obs
    3e5: P, sending_time noise, high bw, high min_rate, small queue_size, long run_dur

    4e1e1: agent: 4, multiply-layer: 7, config.DELTA_SCALE = 0.025, p_reg * 1e0
    4e2: simple sim, P term, 2 obs

    5e1: agent: 5, multiply-layer: 7, config.DELTA_SCALE = 0.025, p_reg * 1e0
    5e2: simple sim, P term, 2 obs

    # WORKING,
        1e1e1 (rewards > 100 after 600 epi, capped at 135 after 2000 epi):
            --python3 train_maddpg.py --max-episode-len 100 --save-rate 100 --num-agents 1 --log-dir 1e3 --num-episodes 30000 --lr 1e-3
        2e1e2, bad until epi 3500 (0.92 rew, negatives before), got 94 rew at epi 4600, 131 rew at epi 5700
            --python3 train_maddpg.py --max-episode-len 100 --save-rate 100 --num-agents 2 --log-dir 2e1e2 --num-episodes 30000 --lr 1e-3

'''

# add to PATHONPATH
import os, sys
from pathlib import Path
cpath = Path(os.getcwd())
sys.path.append(str(cpath.parents[2]))

import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import time
import pickle
import pdb


from maddpg2.maddpg2.trainer.maddpg import MADDPGAgentTrainer # , RandomTrainer
from maddpg2.maddpg2.trainer.network_sim import SimulatedMultAgentNetworkEnv
import tensorflow.contrib.layers as layers
import maddpg2.maddpg2.common.tf_util as U

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--num-agents", type=int, default=1, help="number of good agents")
    parser.add_argument("--log-dir", type=str, default="0", help="directory in which log files are saved")
    parser.add_argument("--debug", type=bool, nargs="?", const=True, default=False, help="Use debugger to track down bad values during training. ")

    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")

    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    #return tf.truncated_normal(shape, stddev=0.1)
    initial = tf.truncated_normal(shape, stddev=0.1)
    #return tf.Variable(initial)
    return tf.compat.v1.get_variable(name='weights', initializer = initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    #return tf.Variable(initial)
    return tf.compat.v1.get_variable(name='bias', initializer=initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.compat.v1.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.compat.v1.summary.scalar('stddev', stddev)
        tf.compat.v1.summary.scalar('max', tf.reduce_max(var))
        tf.compat.v1.summary.scalar('min', tf.reduce_min(var))
        tf.compat.v1.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops."""
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.variable_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        #with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
        #with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.einsum('ij,jk->ik', input_tensor, weights) + biases
            tf.compat.v1.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.compat.v1.summary.histogram('activations', activations)
        return activations

''' add batch_normalization '''
def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm)

        #out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None, normalizer_fn=tf.contrib.layers.batch_norm)

        #out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.softmax)
        #out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.softmax)
        #out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None, normalizer_fn=None)
        return out

''' original achitecture '''
def mlp_model2(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        print("mlp_m2")
        out = input
        print(out)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        print(out)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        print(out)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        print(out)
        return out

''' tensorboard equivalent achitecture '''
def mlp_model3(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    # scope: p_func, q_func, target_q_func
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        #print(input)
        hidden1 = nn_layer(input, input.shape[1].value, num_units, 'layer1')
        #print(hidden1)
        hidden2 = nn_layer(hidden1, hidden1.shape[1].value, num_units, 'layer2')
        #print(hidden2)
        out = nn_layer(hidden2, hidden1.shape[1].value, num_outputs, 'layer3', act=tf.identity)
        #print(out)
        return out

'''attemp to solve the blowing out prob'''
def mlp_model4(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    # scope: p_func, q_func, target_q_func
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        print(input)
        hidden1 = nn_layer(input, input.shape[1].value, num_units, 'layer1')
        print(hidden1)
        hidden2 = nn_layer(hidden1, hidden1.shape[1].value, num_units, 'layer2')
        print(hidden2)
        out = nn_layer(hidden2, hidden1.shape[1].value, num_outputs, 'layer3', act=tf.tanh)
        print(out)
        out = tf.math.multiply(out, 9, 'scale')
        print(out)
        return out

''' run network_sim.py '''
def make_env(arglist):
    # env = SimulatedMultAgentNetworkEnv(arglist, n_features = 2, history_len = 1)
    env = SimulatedMultAgentNetworkEnv(arglist)
    return env

''' run mpe (multi-particle env) '''
def make_env2(scenario_name, arglist, benchmark=False):
    from maddpg2.mpe.multiagent.environment import MultiAgentEnv
    import maddpg2.mpe.multiagent.scenarios as scenarios
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist, sess):
    trainers = []
    p_model = mlp_model3
    q_model = mlp_model3
    trainer = MADDPGAgentTrainer#RandomTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, p_model, q_model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg'), sess=sess))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, p_model, q_model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg'), sess=sess))
    return trainers

def action_check(action_n, obs_n, env, bar):
    action_vals = [action_n[i][0] for i in range(env.n)]
    if (any(abs(action_val) > bar for action_val in action_vals)):
        print(action_vals)
        print(obs_n)



def train(arglist):
    with U.single_threaded_session() as sess:
    #with tf.compat.v1.Session() as sess:
        if(arglist.debug):
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Create environment
        env = make_env(arglist)
        #env = make_env2(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist, sess)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()
        saver = tf.compat.v1.train.Saver(max_to_keep = 10000)

        # Load previous results, if necessary
        if arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            root = os.getcwd()+"/tmp1"
            fname = root + "/model_episode_1200.ckpt"
            U.load_state(fname)


        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info


        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        #best_rew = 0
        #best_params = None

        merged = tf.compat.v1.summary.merge_all()
        init = tf.global_variables_initializer()

        print('Starting iterations...')
        # tf.add_check_numerics_ops()
        while True:
            # get action

            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]

            action_check(action_n, obs_n, env, bar=1000.0)

            # got nan?
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)

            #if(np.sum(rew_n) > best_rew):
            #    best_rew = np.sum(rew_n)
            #    best_params = [agent.parameters for agent in trainers]

            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i in range(env.n):
                obs_n_ = obs_n[i].reshape(3, 10)
                obs_n_ = preprocessing.scale(obs_n_, axis = 0) / 2
                obs_n[i] = obs_n_.reshape(obs_n[i].shape)

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                [agent.reset() for agent in trainers]
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)
                if (loss is not None):
                    # print("update")
                    '''print(len(episode_rewards))
                    print(agent.name)
                    print(loss)
                    print('\n')'''
                #summary_writer.add_summary(loss[0], len(episode_rewards))
                #summary_writer.add_summary(loss[1], len(episode_rewards))

            #sess.run(init)
            #[summary_str] = sess.run([merged])
            #summary_writer.add_summary(summary_str, len(episode_rewards))

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                agent_epi_rew = [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards]
                mean_epi_rew = np.mean(episode_rewards[-arglist.save_rate:])
                num_epi = len(episode_rewards)

                #[summary_str] = sess.run([merged])
                #summary_writer.add_summary(summary_str, num_epi)

                root = os.getcwd()+"/tmp2"
                fname = root + "/model_episode_{}.ckpt".format(len(episode_rewards))
                #save_path = saver.save(sess, fname)
                U.save_state(root, fname, saver=saver)

                #if (mean_epi_rew > best_rew):
                #    best_rew = mean_epi_rew
                #    best_params = [agent.parameters for agent in trainers]

                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, num_epi, mean_epi_rew, agent_epi_rew, round(time.time()-t_start, 3)))
                #print("best_reward: {}, best_params: {}".format(best_rew, best_params))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
