import DagGenerator as dag
import random
import numpy as np

def step(G,attact,defact):
    # immediate reward for both players
    aReward = 0
    dReward = 0
    T = G.graph['horizon']
    #attacker's action
    for attack in attact:
        if isinstance(attack,tuple):
            #check OR node
            aReward += G.edges[attack]['cost']
            if random.uniform(0,1) <= G.edges[attack]['actProb']:
                G.nodes[attack[-1]]['state'] = 1
        else:
            #check AND node
            aReward += G.nodes[attack]['aCost']
            if random.uniform(0,1) <= G.nodes[attack]['actProb']:
                G.nodes[attack]['state'] = 1

    #defender's action
    for node in defact:
        G.nodes[node]['state'] = 0
        dReward += G.nodes[node]['dCost']

    _,targetset = dag.get_Targets(G)
    for node in targetset:
        if G.nodes[node]['state'] == 1:
            aReward += G.nodes[node]['aReward']
            dReward += G.nodes[node]['dPenalty']

    #if goal node prevails for next time step
    # return true state and obs
    return get_att_isActive(G),get_def_hadAlert(G),aReward,dReward



def get_att_isActive(G):
    isActive = []
    for id in np.arange(dag.getNumNodes(G)):
        if G.nodes[id+1]['state'] == 1:
            isActive.append(1)
        else:
            isActive.append(0)
    return isActive



def get_def_hadAlert(G):
    alert = []
    for node in G.nodes:
        if G.nodesp[node]['state'] == 1:
            if random.uniform(0, 1) <= G.nodes[node]['posActiveProb']:
                alert.append(1)
            else:
                alert.append(0)
        elif G.nodesp[node]['state'] == 0:
            if random.uniform(0, 1) <= G.nodes[node]['posInactiveProb']:
                alert.append(1)
            else:
                alert.append(0)
        else:
            raise ValueError("node state is abnormal.")

    return alert


#reset the environment, G_reserved is a copy of the initial env
def reset(G_reserved):
    G = G_reserved.copy()
    return G



