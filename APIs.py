import DagGenerator as dag
import random
import numpy as np

def step(G,attact,defact,targetset,T,timeleft,gamma=1):
    # immediate reward for both players
    aReward = 0
    dReward = 0
    #attacker's action
    for attack in attact:
        if isinstance(attack,tuple):
            #check OR node
            aReward += G.edges[attack]['cost'] * gamma**(T-1-timeleft)
            if random.uniform(0,1) <= G.edges[attack]['actProb']:
                G.nodes[attack[-1]]['state'] = 1
        else:
            #check AND node
            aReward += G.nodes[attack]['aCost']* gamma**(T-1-timeleft)
            if random.uniform(0,1) <= G.nodes[attack]['actProb']:
                G.nodes[attack]['state'] = 1

    #defender's action
    for node in defact:
        G.nodes[node]['state'] = 0
        dReward += G.nodes[node]['dCost']

    for node in targetset:
        if G.nodes[node]['state'] == 1:
            aReward += G.nodes[node]['aReward']* gamma**(T-1-timeleft)
            dReward += G.nodes[node]['dReward']* gamma**(T-1-timeleft)

    #if goal node prevails for next time step
    # return true state and obs
    return get_att_isActive(G),get_def_hadAlert(G),timeleft-1



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
        if G.nodesp[node]['state'] == 0:
            if random.uniform(0, 1) <= G.nodes[node]['posInactiveProb']:
                alert.append(1)
            else:
                alert.append(0)

    return alert



def reset(G_reserved):
    G = G_reserved.copy()
    return G



