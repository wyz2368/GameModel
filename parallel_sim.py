import multiprocessing as mp
import numpy as np
import random
import copy
import time

#TODO: create different copy of env.
#TODO: fix one set of strategies.
#TODO: check [nn]*num_episodes
#TODO: nn should change to mixed strategy
#TODO: API has been changed.
def parallel_sim(env, nn_att, nn_def, num_episodes):
    G_list, att_list, def_list = copy_env(env, num_episodes)
    arg = list(zip(G_list,att_list,[nn_att]*num_episodes,def_list,[nn_def]*num_episodes,[env.T]*num_episodes))
    with mp.Pool() as pool:
        r = pool.map_async(single_sim, arg)
        a =r.get()
    return np.sum(np.array(a),0)/num_episodes


def single_sim(param): #single for single episode.
    # TODO: APIs have been changed.
    aReward = 0
    dReward = 0

    G, attacker, nn_att, defender, nn_def, T = param

    for t in range(T):
        timeleft = T - t
        attacker.att_greedy_action_builder_single(G, timeleft, nn_att)
        att_action_set = attacker.attact
        defender.def_greedy_action_builder_single(G, timeleft, nn_def)
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
    env.reset_everything()
    for _ in np.arange(num_episodes):
        G_list.append(env.G_reserved.copy())
        att_list.append(copy.deepcopy(env.attacker))
        def_list.append(copy.deepcopy(env.defender))

    return G_list, att_list, def_list

