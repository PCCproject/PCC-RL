from network_sim import SimulatedNetworkEnv
import numpy as np
import json

def run_episode(env, parameters, log=False):  
    observation = env.reset()
    totalreward = 0
    all_obs = []
    all_rews = []
    for i in range(0, 200):
        observation[0] /= 1000
        observation[1] /= 100
        observation[2] /= 100
        observation[3] /= 100
        action = 0.95 if np.matmul(parameters,observation) < 0 else 1.05
        observation, reward, done, info = env.step([action])
        if log:
            all_obs.append(np.copy(observation))
            all_rews.append(np.copy(reward))
        totalreward += reward
        if done:
            break
    if log:
        return all_obs, all_rews
    return totalreward / 200

bestparams = []
bestreward = -1e12
env = SimulatedNetworkEnv()
for i in range(0, 50):
    runparameters = np.random.rand(6) * 2 - 1
    runreward = -1e12
    for j in range(0, 100):
        parameters = runparameters + 0.1 * (np.random.rand(6) * 2 - 1)
        reward = run_episode(env,parameters)
        if reward > runreward:
            runreward = reward
            runparameters = parameters
    
    if runreward > bestreward:
        bestreward = runreward
        bestparams.append(runparameters)
        print("Rew = %f" % runreward)
    print("Finished model %d, final reward %f" % (i, runreward))

for p in range(0, len(bestparams)):
    params = bestparams[p]
    all_obs, all_rews = run_episode(env, params, True)
    data = []
    for i in range(0, len(all_obs)):
        ob = all_obs[i]
        rew = all_rews[i]
        data.append({"Name":"Calculate Utility", "MI Start Time":i, "Target Rate":float(ob[0]), "Utility":float(rew), "Avg RTT":float(ob[4]), "Loss Rate":float(ob[3])})
    with open("logs/log_%d.json" % p, "w") as l:
        json.dump({"Events":data}, l, indent=4)
