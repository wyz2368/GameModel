import APIs
import DagGenerator as dag
import numpy as np
import random

def att_greedy_action_builder(G,attObs,nn_att,A):
    attackSet = set()
    isDup = 0
    isLegal = 1
    # pass is indicated by -1
    x = 0
    #Do we really need pass?
    #The answer is no in this version
    while not isDup and isLegal and x != -1 and random.uniform(0,1) > 0.1:
        att_input = att_obs_constructor(attObs,attackSet)
        x = nn_att(att_input)
        isDup = (x in attackSet)
        if isinstance(x,tuple):
            isLegal = (x == -1 or x[0] in A)
        else:
            isLegal = (x == -1 or dag.predecessors(G,x) in A)
        if not isDup and isLegal and x != -1:
            attackSet.add(x)
    return attackSet


def def_greedy_action_builder(defObs,nn_def):
    defendSet = set()
    isDup = 0
    # pass is indicated by -1
    x = 0
    while not isDup and x != -1 and random.uniform(0,1) > 0.1:
        def_input = def_greedy_action_builder(defObs,defendSet)
        x = nn_def(def_input)
        isDup = (x in defendSet)
        if not isDup and x != -1:
            defendSet.add(x)
    return defendSet



def att_obs_constructor(G,trueobs,attackSet,timeleft):
    canAttack, inAttackSet = get_att_canAttack_inAttackSet(G, attackSet)
    att_input = trueobs+canAttack+inAttackSet+[timeleft]
    return att_input

def def_obs_constructor(G,obs,defendSet,previous_obs,timeleft,defact_tm1):
    wasdef = get_def_wasDefended(G,defact_tm1)
    indef = get_def_inDefenseSet(G,defendSet)
    # no need for repeating timeleft, so it is not N
    def_input = previous_obs+obs+wasdef+indef+[timeleft]
    return def_input


def get_att_canAttack(G):
    canAttack = []
    _, Andnodeset = dag.get_ANDnodes(G)
    for andnode in Andnodeset:
        precondflag = 1
        precond = dag.predecessors(G, andnode)
        for prenode in precond:
            if G.nodes[prenode]['state'] == 0:
                precondflag = 0
                break
        if G.nodes[andnode]['state'] == 0 and precondflag:
            canAttack.append(1)
        else:
            canAttack.append(0)

    oredgeset = dag.get_ORedges(G)
    for (father, son) in oredgeset:
        if G.nodes[father]['state'] == 1 and G.nodes[son]['state'] == 0:
            canAttack.append(1)
        else:
            canAttack.append(0)

    return canAttack

def get_att_canAttack_inAttackSet(G, attact):
    canAttack = []
    inAttackSet = []
    _, Andnodeset = dag.get_ANDnodes(G)
    for andnode in Andnodeset:
        if andnode in attact:
            inAttackSet.append(1)
        else:
            inAttackSet.append(0)
        precondflag = 1
        precond = dag.predecessors(G, andnode)
        for prenode in precond:
            if G.nodes[prenode]['state'] == 0:
                precondflag = 0
                break
        if G.nodes[andnode]['state'] == 0 and precondflag:
            canAttack.append(1)
        else:
            canAttack.append(0)

    oredgeset = dag.get_ORedges(G)
    for (father, son) in oredgeset:
        if (father, son) in attact:
            inAttackSet.append(1)
        else:
            inAttackSet.append(0)
        if G.nodes[father]['state'] == 1 and G.nodes[son]['state'] == 0:
            canAttack.append(1)
        else:
            canAttack.append(0)

    return canAttack, inAttackSet

def get_def_wasDefended(G,defact_tm1):
    wasdef = []
    for node in G.nodes:
        if node in defact_tm1:
            wasdef.append(1)
        else:
            wasdef.append(0)
    return wasdef


def get_def_inDefenseSet(G,defact):
    indef = []
    for node in G.nodes:
        if node in defact:
            indef.append(1)
        else:
            indef.append(0)
    return indef