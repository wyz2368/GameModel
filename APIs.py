import DagGenerator as dag
import random

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
    # construct attacker's obs(true state)


    # construct defender's obs

def get_att_isActive(G):



def get_att_canAttack(G):



def get_att_inAttackSet(G):



def get_def_hadAlert(G):



def get_def_wasDefended(G):



def get_def_inDefenseSet(G):



def reset(G):
    print()



