import random
import numpy as np
import multiprocessing as mp
import copy
import time
import operator

def rand_strategies_payoff(env, num_episodes):
    aReward_list = np.array([])
    dReward_list = np.array([])
    num_resource_def = 2
    num_resource_att = 2

    tl = []
    t1 = time.time()
    for i in range(num_episodes): #can be run parallel
        aReward = 0
        dReward = 0
        env.reset_everything()

        for t in range(env.T):
            att_action_set = env.attacker.uniform_strategy(env.G, num_resource_att)
            def_action_set = set(sorted(random.sample(list(env.G.nodes), min(num_resource_def,env.G.number_of_nodes()))))
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
        t3 = time.time()
        aReward_list = np.append(aReward_list,aReward)
        dReward_list = np.append(dReward_list,dReward)
        t4 = time.time()
        tl.append(t4-t3)
    t2 = time.time()
    return np.mean(aReward_list), np.mean(dReward_list), t2-t1, sum(tl)


# def parallel_sim(env,  num_episodes):
#     aReward_list = np.array([])
#     dReward_list = np.array([])
#
#     G_list, att_list = copy_env(env, num_episodes)
#
#     with mp.Pool() as pool:
#         for i in np.arange(num_episodes):
#             r = pool.apply_async(rand_single_sim_parallel,(G_list[i], att_list[i], env.T))
#             aReward_list = np.append(aReward_list,r.get()[0])
#             dReward_list = np.append(dReward_list,r.get()[1])
#
#     return np.mean(aReward_list), np.mean(dReward_list)


def rand_parallel_sim(env,  num_episodes):
    G_list, att_list = copy_env(env, num_episodes)

    arg = list(zip(G_list, att_list, [env.T]*num_episodes))


    with mp.Pool() as pool:
        r = pool.map_async(rand_single_sim_parallel,arg)
        a = r.get()
    return np.sum(np.array(a),0)/num_episodes

# def rand_single_sim_parallel(G, attacker, T):
def rand_single_sim_parallel(param):
    num_resource_def = 2
    num_resource_att = 2
    G, attacker, T = param
    # G = env.G
    # attacker = env.attacker
    # T = env.T

    aReward = 0
    dReward = 0

    for t in range(T):
        att_action_set = attacker.uniform_strategy(G, num_resource_att)
        def_action_set = set(sorted(random.sample(list(G.nodes), min(num_resource_def,G.number_of_nodes()))))
        for attack in att_action_set:
            if isinstance(attack, tuple):
                # check OR node
                aReward += G.edges[attack]['cost']
                if random.uniform(0, 1) <= G.edges[attack]['actProb']:
                    G.nodes[attack[-1]]['state'] = 1
            else:
                # check AND node
                aReward += G.nodes[attack]['aCost']
                if random.uniform(0, 1) <= G.nodes[attack]['actProb']:
                    G.nodes[attack]['state'] = 1
        # defender's action
        for node in def_action_set:
            G.nodes[node]['state'] = 0
            dReward += G.nodes[node]['dCost']
        _, targetset = get_Targets(G)
        for node in targetset:
            if G.nodes[node]['state'] == 1:
                aReward += G.nodes[node]['aReward']
                dReward += G.nodes[node]['dPenalty']


    return aReward, dReward



def get_Targets(G):
    count = 0
    targetset = set()
    for node in G.nodes:
        if G.nodes[node]['type'] == 1:
            count += 1
            targetset.add(node)
    return count,targetset

def copy_env(env, num_episodes):
    G_list = []
    att_list = []
    for _ in np.arange(num_episodes):
        G_list.append(env.G_reserved.copy())
        att_list.append(copy.deepcopy(env.attacker))

    return G_list, att_list


