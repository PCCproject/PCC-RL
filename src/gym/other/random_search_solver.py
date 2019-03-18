from network_sim import SimulatedNetworkEnv
import numpy as np
import json

def run_episode(env, parameters, log=False):  
    observation = env.reset()
    totalreward = 0
    all_obs = []
    all_rews = []
    for i in range(0, 200):
        action = 0.95 if np.matmul(parameters,observation) < 0 else 1.05
        observation, reward, done, info = env.step([action])
        if log:
            all_obs.append(observation)
            all_rews.append(reward)
        totalreward += reward
        if done:
            break
    if log:
        return all_obs, all_rews
    return totalreward / 200

bestparams = []
bestreward = -1e12
env = SimulatedNetworkEnv()
for i in range(0, 5000):
    if (i % 10) == 0:
        print("Trial %d/1000" % i)
    parameters = np.random.rand(3) * 2 - 1
    reward = run_episode(env,parameters)
    if reward > bestreward:
        bestreward = reward
        bestparams.append(parameters)
        print("Rew = %f" % reward)

for p in range(0, len(bestparams)):
    params = bestparams[p]
    all_obs, all_rews = run_episode(env, params, True)
    data = []
    for i in range(0, len(all_obs)):
        ob = all_obs[i]
        rew = all_rews[i]
        data.append({"Name":"Calculate Utility", "MI Start Time":i, "Target Rate":ob[0], "Utility":rew, "Avg RTT":ob[4], "Loss Rate":ob[3]})
    with open("logs/log_%d.json" % p, "w") as l:
        json.dump({"Events":data}, l)
