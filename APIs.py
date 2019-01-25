import DagGenerator as dag
import random

def step(G,attact,defact,targetset):
    # immediate reward for both players
    aReward = 0
    dReward = 0
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

    for node in targetset:
        if G.nodes[node]['state'] == 1:
            aReward += G.nodes[node]['aReward']
            dReward += G.nodes[node]['dReward']

    #if goal node prevails for next time step



def reset(G):
    print()



