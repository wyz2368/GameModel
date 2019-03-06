import multiprocessing as mp
import numpy as np
import random
import time

#TODO: create different copy of env.

def parallel_sim(env, nn_att_list, nn_def_list, num_episodes):
    aReward_list = np.array([])
    dReward_list = np.array([])
    G_list, att_list, def_list = copy_env(env, num_episodes)
    with mp.Pool() as pool:
        for i in np.arange(num_episodes):
            r = pool.apply_async(single_sim(G_list[i],att_list[i],nn_att_list[i],def_list[i],nn_def_list[i],env.T))
            aReward_list = np.append(aReward_list,r.get()[0])
            dReward_list = np.append(dReward_list,r.get()[1])
    return np.mean(aReward_list), np.mean(dReward_list)


def single_sim(G, attacker, nn_att, defender, nn_def, T): #single for single episode.
    # TODO: APIs have been changed.
    aReward = 0
    dReward = 0
    G.reset()
    attacker.reset_att()
    defender.reset_def()
    for t in range(T):
        timeleft = T - t
        attacker.att_greedy_action_builder(G, timeleft, nn_att)
        att_action_set = attacker.attact
        defender.def_greedy_action_builder(G, timeleft, nn_def)
        def_action_set = defender.defact
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
    def_list = []
    for _ in np.arange(num_episodes):
        G_list.append(env.G.copy())
        att_list.append(env.attacker.copy())
        def_list.append(env.defender.copy())

    return G_list, att_list, def_list