import networkx as nx
import numpy as np

num_attr_N = 12
num_attr_E = 4

def daggenerator_wo_attrs(nodeset,edgeset):
    G = nx.DiGraph()
    G.add_nodes_from(nodeset,
                     root = 0, #0:NONROOT 1:ROOTNODE
                     type = 0,# 0:NONTARGET 1:TARGET
                     eType = 0,# 0:OR 1:AND node
                     state = 0,# 0:Inactive 1:Active
                     aReward = 0.0,
                     dPenalty = 0.0,
                     dCost = 0.0,
                     aCost = 0.0,
                     posActiveProb = 1.0, #prob of sending positive signal if node is active
                     posInactiveProb = 0.0, #prob of sending positive signal if node is inactive
                     actProb = 1.0, #prob of becoming active if being activated, for AND node only
                     topoPosition = -1)

    G.add_edges_from(edgeset,
                     type = 0, #0:NORMAL 1:VIRTUAL
                     cost = 0, # Cost for attacker on OR node
                     weight = 0.0,
                     actProb=1.0) # probability of successfully activating, for OR node only

    return G


def isProb(p):
    return p >= 0.0 and p <= 1.0

# Node Operations
# Get Info
def isOrType_N(G,id):
    return G.nodes[id]['type'] == 0

def getState_N(G,id):
    return G.nodes[id]['state']

def getType_N(G,id):
    return G.nodes[id]['type']

def getActivationType_N(G,id):
    return G.nodes[id]['eType']

def getAReward_N(G,id):
    return G.nodes[id]['aReward']

def getDPenalty_N(G,id):
    return G.nodes[id]['dPenalty']

def getDCost_N(G,id):
    return G.nodes[id]['dCost']

def getACost_N(G,id):
    return G.nodes[id]['aCost']

def getActProb_N(G,id):
    return G.nodes[id]['actProb']

def getTopoPosition_N(G,id):
    return G.nodes[id]['topoPosition']

def getposActiveProb_N(G,id):
    return G.nodes[id]['posActiveProb']

def getposInactiveProb_N(G,id):
    return G.nodes[id]['posInactiveProb']

# Set Info

def setState_N(G,id,value):
     G.nodes[id]['state'] = value

def setType_N(G,id,value):
     G.nodes[id]['type'] = value

def setActivationType_N(G,id,value):
     G.nodes[id]['eType'] = value

def setAReward_N(G,id,value):
     G.nodes[id]['aReward'] = value

def setDPenalty_N(G,id,value):
     G.nodes[id]['dPenalty'] = value

def setDCost_N(G,id,value):
     G.nodes[id]['dCost'] = value

def setACost_N(G,id,value):
     G.nodes[id]['aCost'] = value

def setActProb_N(G,id,value):
     G.nodes[id]['actProb'] = value

def setTopoPosition_N(G,id,value):
     G.nodes[id]['topoPosition'] = value

def setposActiveProb_N(G,id,value):
     G.nodes[id]['posActiveProb'] = value

def setposInactiveProb_N(G,id,value):
     G.nodes[id]['posInactiveProb'] = value


# Edge Operations
# Get Info

def getType_E(G,edge):
    return G.edges[edge]['type']

def getACost_E(G,edge):
    return G.edges[edge]['cost']

def getActProb_E(G,edge):
    return G.edges[edge]['actProb']

def getweight_E(G,edge):
    return G.edges[edge]['weight']

# Set Info

def setType_E(G,edge,value):
     G.edges[edge]['type'] = value

def setACost_E(G,edge,value):
     G.edges[edge]['cost'] = value

def setActProb_E(G,edge,value):
     G.edges[edge]['actProb'] = value

def setweight_E(G,edge,value):
     G.edges[edge]['weight'] = value

# Print Info
def print_N(G,id):
    print(G.nodes[id])

def print_E(G,edge):
    print(G.edges[edge])

# Other Operations
def getNumNodes(G):
    return G.number_of_nodes()

def getNumEdges(G):
    return G.number_of_edges()

def inDegree(G,id):
    return G.in_degree(id)

def outDegree(G,id):
    return G.out_degree(id)

def predecessors(G,id):
    return set(G.predecessors(id))

def successors(G,id):
    return set(G.sucessors(id))

def isDAG(G):
    return nx.is_directed_acyclic_graph(G)

def getEdges(G):
    return G.edges()

def get_num_ANDnodes(G):
    count = 0
    for node in G.nodes:
        if G.nodes[node]['eType'] == 1:
            count += 1
    return count

def get_num_Targets(G):
    count = 0
    for node in G.nodes:
        if G.nodes[node]['type'] == 1:
            count += 1
    return count

def get_num_Roots(G):
    count = 0
    for node in G.nodes:
        if G.nodes[node]['root'] == 1:
            count += 1
    return count

def get_num_NormalEdges(G):
    count = 0
    for edge in G.edges:
        if G.edges[edge]['type'] == 0:
            count += 1
    return count



# test
# G = daggenerator_wo_attrs([1,2,3],[(1,2),(1,3)])
# print(G.nodes.data())
# print(G.edges.data())

# Attributes Initialization
def assignAttr_N(G,id,attr): #add code to check the lenth match
    G.nodes[id].update(dict(zip(G.nodes[id].keys(),attr)))

def assignAttr_E(G,edge,attr):
    G.edges[edge].update(dict(zip(G.nodes[id].keys(),attr)))

def attrGenerator(num_nodes,num_edges,num_attr_N = 12,num_attr_E = 4,num_targets = 1,num_root = 1):
    # Hard coding
    #











