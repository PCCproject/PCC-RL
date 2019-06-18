 # examples/quickstart.py

import numpy as np

from tensorforce.agents import PPOAgent
from tensorforce.agents import TRPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

# Create an OpenAIgym environment
#env = OpenAIGym('CartPole-v0', visualize=False)
env = OpenAIGym('PccNs-v0', visualize=False)

# Network as list of layers
network_spec = [
    dict(type='dense', size=32, activation='relu'),
    dict(type='internal_lstm', size=32)#,
    #dict(type='dense', size=32, activation='relu')
    #dict(type='dense', size=32, activation='tanh'),
    #dict(type='dense', size=32, activation='tanh')
]

#"""
agent = TRPOAgent(
    states=env.states,
    actions=env.actions,
    network=network_spec,
    # Model
    scope='trpo',
    discount=0.99,#)
    # DistributionModel
    #distributions_spec=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode=None,
    baseline=None,
    baseline_optimizer=None,
    gae_lambda=None,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    #summary_spec=None#,
    #distributed_spec=None
    batching_capacity=2048,
    update_mode={"unit":"episodes", "batch_size":40}
)
#"""
"""
agent = PPOAgent(
    states=env.states,
    actions=env.actions,
    network=network_spec,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    optimization_steps=10,
    # Model
    scope='ppo',
    discount=0.99,#)
    # DistributionModel
    #distributions_spec=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode=None,
    baseline=None,
    baseline_optimizer=None,
    gae_lambda=None,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    #summary_spec=None#,
    #distributed_spec=None
    batching_capacity=2048,
    update_mode={"unit":"episodes", "batch_size":40}
)
#"""

# Create the runner
runner = Runner(agent=agent, environment=env)

ewma = 0.0

# Callback function printing episode statistics
def episode_finished(r):
    global ewma
    ewma *= 0.99
    ewma += 0.01 * r.episode_rewards[-1]
    print("Finished episode {ep} after {ts} timesteps (ewma rew: {ewma}, reward: {reward})".format(ep=r.episode, ts=r.episode_timestep, ewma=ewma,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=9600, max_episode_timesteps=400, episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
