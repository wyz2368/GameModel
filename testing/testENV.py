import DagGenerator as dag

env = dag.Environment(numNodes=5, numEdges=4, numRoot=2, numGoals=1)

nodeset = [1,2,3,4,5]
edgeset = [(1,2),(2,3),(2,4),(5,2)]

attr = {}
attr['nodes'] = nodeset
attr['edges'] = edgeset
attr['Nroots'] = [1,0,0,0,1]
attr['Ntypes'] = [0,0,0,1,0]
attr['NeTypes'] = [1,1,0,0,1]
attr['Nstates'] = [0,0,0,0,0]
attr['NaRewards'] = [0,0,0,3,0]
attr['NdPenalties'] = [0,0,0,-3,0]
attr['NdCosts'] = [-1,-1,-1,-1,-1]
attr['NaCosts'] = [-1,-1,-1,-1,-1]
attr['NposActiveProbs'] = [0.6,0.6,0.6,0.6,0.6]
attr['NposInactiveProbs'] =[0.2,0.2,0.2,0.2,0.2]
attr['NactProbs'] = [0.8,0.8,0.8,0.8,0.8]

attr['Eeids'] = [1,2,3,4]
attr['Etypes'] = [0,0,0,0]
attr['actProb'] = [0,0.9,0.9,0]
attr['Ecosts'] = [0,-1,-1,0]


# nodeset = [1,2]
# edgeset = [(1,2)]

env.daggenerator_wo_attrs(nodeset, edgeset)
env.specifiedDAG(attr)
env.save_graph_copy()

# env.visualize()

# print(env.G.nodes.data()[4])
# print(env.G.edges.data())

env.create_players()

#test attacker
# print(env.attacker.ORedges)
# print(env.attacker.ANDnodes)
# print(env.attacker.actionspace)
# print(env.attacker.get_att_canAttack_inAttackSet(env.G))
# print(env.attacker.uniform_strategy(env.G,1))
# env.attacker.update_canAttack(env.attacker.get_att_canAttack(env.G))
# print(env.attacker.canAttack)
# env.attacker.reset_att()
# print(env.attacker.canAttack)

#test defender
# print(env.defender.num_nodes)
# print(env.defender.observation)
# print(env.defender.history)
# print(env.defender.prev_obs)
# print(env.defender.defact)
# print(env.defender.prev_defact)
# print(env.defender.rand_limit)

# env.defender.defact.add(2)
# env.defender.defact.add(3)
# env.defender.defact.add(5)
#
# print(env.defender.get_def_wasDefended(env.G))
# print(env.defender.get_def_inDefenseSet(env.G))
# print(env.defender.get_def_actionspace(env.G))
# print(env.defender.uniform_strategy(env.G))
#
# env.defender.update_obs([0,0,0,0,1])
# env.defender.update_obs([0,0,0,1,1])
# print(env.defender.observation)
#
# env.defender.save_defact2prev()
#
# print('*******')
# print(env.defender.observation)
# print(env.defender.prev_obs)
# print(env.defender.defact)
# print(env.defender.prev_defact)
#
# print(env.defender.def_obs_constructor(env.G,9))


#test the environment
# a = [(1,2),(5,9),(4,3),(1,9),(2,3)]
# print(env.sortEdge(a))
# print(env.getHorizon_G())
# print(env.G.nodes.data())
# print(env.isOrType_N(5))
# print(env.G.nodes)
# for i in env.G.nodes:
    # print(env.getState_N(i))
    # print(env.getType_N(i))
    # print(env.getActivationType_N(i))
    # print(env.getAReward_N(i))
    # print(env.getDPenalty_N(i))
    # print(env.getDCost_N(i))
    # print(env.getACost_N(i))
    # print(env.getActProb_N(i))
    # print(env.getposActiveProb_N(i))
    # print(env.getposInactiveProb_N(i))

# print(env.G.edges)
# for i in env.G.edges:
#     # print(env.getid_E(i))
#     print(env.getActProb_E(i))

# env.print_N(1)
# env.print_E((2,3))

# print(env.getNumNodes())
# print(env.getNumEdges())
# for i in env.G.nodes:
    # print(env.inDegree(i))
    # print(env.outDegree(i))
    # print(env.predecessors(i))
    # print(env.successors(i))

# print(env.isDAG())
# print(env.getEdges())
# print(env.get_ANDnodes())
# print(env.get_ORnodes())
# print(env.get_ORedges())
# print(env.get_Targets())
# print(env.get_Roots())
# print(env.get_NormalEdges())
# print(env.get_att_isActive())
# print(env.get_def_hadAlert())
# print(env.get_att_actionspace())
# print(env.get_def_actionspace())

# a = [1,2,3]
# print(env.check_nodes_sorted(a))

# test step functions


print(env.G_reserved.nodes)













