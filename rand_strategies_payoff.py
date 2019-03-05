import random
import numpy as np

def rand_strategies_payoff(env, num_episodes):
    aReward_list = np.array([])
    dReward_list = np.array([])
    num_resource_def = 4
    num_resource_att = 4

    for i in range(num_episodes): #can be run parallel
        aReward = 0
        dReward = 0
        env.G.reset()
        env.attacker.reset_att()
        for t in range(env.T):
            att_action_set = env.attacker.uniform_strategy(env.G, num_resource_att)
            def_action_set = set(sorted(random.sample(list(env.G.nodes), num_resource_def)))
            for attack in att_action_set:
                if isinstance(attack, tuple):
                    # check OR node
                    aReward += env.G.edges[attack]['cost']
                    if random.uniform(0, 1) <= env.G.edges[attack]['actProb']:
                        env.G.nodes[attack[-1]]['state'] = 1
                else:
                    # check AND node
                    aReward += env.G.nodes[attack]['aCost']
                    if random.uniform(0, 1) <= env.G.nodes[attack]['actProb']:
                        env.G.nodes[attack]['state'] = 1
            # defender's action
            for node in def_action_set:
                env.G.nodes[node]['state'] = 0
                dReward += env.G.nodes[node]['dCost']
            _, targetset = env.get_Targets()
            for node in targetset:
                if env.G.nodes[node]['state'] == 1:
                    aReward += env.G.nodes[node]['aReward']
                    dReward += env.G.nodes[node]['dPenalty']

        aReward_list = np.append(aReward_list,aReward)
        dReward_list = np.append(dReward_list,dReward)

    return np.mean(aReward_list), np.mean(dReward_list)



