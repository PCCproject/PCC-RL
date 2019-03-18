from network_sim import SimulatedNetworkEnv
import numpy as np
import json

bias = 4.0

def run_episode(env, parameters, log=False):
    observation = env.reset()
    totalreward = 0
    all_obs = []
    all_rews = []
    for i in range(0, 400):
        #observation[0] /= 1000
        #observation[1] /= 1000
        #observation[2] /= 1
        #observation[3] /= 1
        ##observation[4] /= 1
        #observation[5] /= 1000
        #observation[6] /= 1000 * 1500 * 2
        #observation[7] /= 1
        #observation[8] /= 1000
        action = np.matmul(parameters, observation) + bias
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

params = np.array([
#    0.0, # Send Rate
#    0.0, # Recv Rate
#    0.0, # Latency
#    -1.0, # Loss
     -50.0, # Latency Inflation
     -1.0, # Latency Ratio
#    # Bytes In Flight
     -2.0 # Send Ratio
#    0.0 # Reward
])


env = SimulatedNetworkEnv()
rews = []
for i in range(0, 110):
    all_obs, all_rews = run_episode(env, params, True)
    data = []
    rews.append(np.sum(all_rews))
    #print("Reward: %f" % np.sum(all_rews))
    """
    for i in range(0, len(all_obs)):
        ob = all_obs[i]
        rew = all_rews[i]
        data.append({"Name":"Step", "Time":i, "Send Rate":float(ob[0]), "Throughput":float(ob[1]), "Reward":float(rew), "Latency":float(ob[2]), "Loss Rate":float(ob[3])})
    with open("logs/log_%d.json" % p, "w") as l:
        json.dump({"Events":data}, l, indent=4)
    #"""
print("Reward: %f" % np.mean(rews))
