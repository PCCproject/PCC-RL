import sys
import numpy as np
import random
import tensorflow as tf
import maddpg2.maddpg2.common.tf_util as U

from maddpg2.maddpg2.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def gaussian_noise_layer(input_layer, mean, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=mean, stddev=std, dtype=tf.float32)
    return input_layer + noise

def p_train2(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None, sess=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions

        # act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n] #

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        # act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        act_ph_n = [tf.placeholder(dtype=tf.float32, shape=[None]+[1], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        U.scope_vars(U.absolute_scope_name("p_func"))

        p = p_func(p_input, 1, scope="p_func", num_units=num_units)
        # p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)

        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        # act_pd = act_pdtype_n[p_index].pdfromflat(p)
        act_pd = p

        # act_sample = act_pd.sample()
        # act_sample = act_pd
        act_sample = gaussian_noise_layer(act_pd, 0.0, 0.5)

        # p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
        p_reg = tf.reduce_mean(tf.square(act_pd))


        act_input_n = act_ph_n + []

        # act_input_n[p_index] = act_pd.sample()
        # act_input_n[p_index] = act_pd
        act_input_n[p_index] = gaussian_noise_layer(act_pd, 0.0, 0.5)

        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        #q_input = tf.clip_by_value(q_input, -1000000.0, 1000000.0)
        print("\n\n")
        print("p_train")
        print(p_input)
        print(obs_ph_n + act_input_n)
        print(q_input)
        print("\n\n")
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)

        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        #print(grad_norm_clipping)
        #print(optimize_expr)
        #print_op = tf.print("Debug output:", optimize_expr, "\n", output_stream=sys.stdout)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr], sess=sess)
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample, sess=None)
        p_values = U.function([obs_ph_n[p_index]], p, sess=None)

        # target network
        # target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p = p_func(p_input, 1, scope="target_p_func", num_units=num_units)

        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        # target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        # target_act_sample = act_pd
        target_act_sample = gaussian_noise_layer(act_pd, 0.0, 0.5)

        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample, sess=None)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train2(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64, sess=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders, global observation and global action, i.e. 30, 30, 30, 1, 1, 1
        obs_ph_n = make_obs_ph_n

        # act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        act_ph_n = [tf.placeholder(dtype=tf.float32, shape=[None]+[1], name="action"+str(i)) for i in range(len(act_space_n))]

        target_ph = tf.placeholder(tf.float32, [None], name="target")


        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        #q_input = tf.clip_by_value(q_input, -1000000.0, 1000000.0)
        print("\n\n")
        print("q_train")
        print(obs_ph_n + act_ph_n)
        print(q_input)
        print("\n\n")

        # local observation and action i.e. 30, 1
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)

        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]

        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr], sess=sess)
        q_values = U.function(obs_ph_n + act_ph_n, q, sess=None)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q, sess=None)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None, sess=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        print("create distrbutions")
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        print("finsih create dist")

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        U.scope_vars(U.absolute_scope_name("p_func"))

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)

        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)
        print(act_pd)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam())) #???
        # p_reg = tf.reduce_mean(tf.square(act_sample))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()

        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        #q_input = tf.clip_by_value(q_input, -1000000.0, 1000000.0)
        print("\n\n")
        print("p_train")
        print(obs_ph_n + act_input_n)
        print(q_input)
        print("\n\n")
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        #loss = tf.add(print_out, p_reg * 1e-3)
        #loss = print_out + p_reg * 1e-3
        # loss = pg_loss + p_reg * 1e-3
        loss = pg_loss + p_reg * 1e-1

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        #print(grad_norm_clipping)
        #print(optimize_expr)
        print_op = tf.print("Debug output:", optimize_expr, "\n", output_stream=sys.stdout)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr], sess=sess)
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample, sess=None)
        p_values = U.function([obs_ph_n[p_index]], p, sess=None)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample, sess=None)

        with tf.control_dependencies([print_op]):
            return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64, sess=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders, global observation and global action, i.e. 30, 30, 30, 1, 1, 1
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")


        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        #q_input = tf.clip_by_value(q_input, -1000000.0, 1000000.0)
        print("\n\n")
        print("q_train")
        print(obs_ph_n + act_ph_n)
        print(q_input)
        print("\n\n")

        # local observation and action i.e. 30, 1
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]

        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr], sess=sess)
        q_values = U.function(obs_ph_n + act_ph_n, q, sess=None)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q, sess=None)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, p_model, q_model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False, sess=None):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.sess = sess
        obs_ph_n = []

        ##########---->

        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=q_model,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            sess=self.sess
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=p_model,
            q_func=q_model,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            sess=self.sess
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        print(args.batch_size)
        print(args.max_episode_len)
        self.replay_sample_index = None
        self.parameters = None

    def reset(self):
        pass

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    ''' clear indices of repley buffer'''
    def preupdate(self):
        self.replay_sample_index = None

    ''' nan happens after a few updates'''
    def update(self, agents, t):
        # self.max_replay_buffer_len = 25600   1024 * 100 / 12
        if len(self.replay_buffer) < (self.max_replay_buffer_len/12): # replay buffer is not large enough
            print("{}, {}".format(len(self.replay_buffer), self.max_replay_buffer_len/12))
            return
        if not t % 400 == 0:  # only update every 400 steps, that is 4 episodes, because --max-episode-len = 100
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()
        '''
        if(abs(q_loss) > 15322843000.0):
            tf.print(*(obs_n + act_n + [target_q]))

        if(abs(p_loss) > 15322843000.0):
            tf.print(*(obs_n + act_n))
            '''

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]

'''
class RandomTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args

        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
        self.parameters = parameters = np.random.rand(30) * 2 - 1

    def reset(self):
        pass
        #self.parameters = np.random.rand(30) * 2 - 1

    def action(self, obs):
        self.parameters = np.random.rand(30) * 2 - 1
        ret = np.matmul(self.parameters, obs)
        return [ret]

    def preupdate(self):
        pass

    def update(self, agents, t):
        return []

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))
'''
