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

# print(env.G.nodes.data()[4])
# print(env.G.edges.data())

env.create_players()

#test attacker
print(env.attacker.ORedges)
print(env.attacker.ANDnodes)
print(env.attacker.actionspace)
print(env.attacker.get_att_canAttack_inAttackSet(env.G))
print(env.attacker.uniform_strategy(env.G,1))
env.attacker.update_canAttack(env.attacker.get_att_canAttack(env.G))
print(env.attacker.canAttack)
#test defender
