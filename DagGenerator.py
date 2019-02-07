import networkx as nx
import numpy as np
import sys

class Environment(object):

    def __init__(self, num_attr_N=12, num_attr_E=5):
        self.num_attr_N = num_attr_N
        self.num_attr_E = num_attr_E
        self.G = nx.DiGraph()

    def daggenerator_wo_attrs(self, nodeset,edgeset,T,graphid):
        self.G = nx.DiGraph(horizon = T, id = graphid)
        self.G.add_nodes_from(nodeset,
                         root = 0, #0:NONROOT 1:ROOTNODE
                         type = 0,# 0:NONTARGET 1:TARGET
                         eType = 0,# 0:OR 1:AND node
                         state = 0,# 0:Inactive 1:Active
                         aReward = 0.0, # GREATER THAN OR EQUAL TO 0
                         dPenalty = 0.0, # SAME^
                         dCost = 0.0, # SAME^
                         aCost = 0.0, # SAME^
                         posActiveProb = 1.0, #prob of sending positive signal if node is active
                         posInactiveProb = 0.0, #prob of sending positive signal if node is inactive
                         actProb = 1.0, #prob of becoming active if being activated, for AND node only
                         topoPosition = -1)
        self.G.add_edges_from(edgeset,
                         eid = -1,
                         type = 0, #0:NORMAL 1:VIRTUAL
                         cost = 0, # Cost for attacker on OR node, GREATER THAN OR EQUAL TO 0
                         weight = 0.0,
                         actProb=1.0) # probability of successfully activating, for OR node only
        return self.G

    def isProb(self,p):
        return p >= 0.0 and p <= 1.0

    def sortEdge(self,edgeset):
        sorted_by_first_second = sorted(edgeset, key=lambda tup: (tup[0], tup[1]))
        return sorted_by_first_second

    # Graph Operation
    def getHorizon_G(self):
        return self.G.graph['horizon']

    def setHorizon_G(self,value):
        self.G.graph['horizon'] = value

    # Node Operations
    # Get Info
    def isOrType_N(self,id):
        return self.G.nodes[id]['type'] == 0

    def getState_N(self,id):
        return self.G.nodes[id]['state']

    def getType_N(self,id):
        return self.G.nodes[id]['type']

    def getActivationType_N(self,id):
        return self.G.nodes[id]['eType']

    def getAReward_N(self,id):
        return self.G.nodes[id]['aReward']

    def getDPenalty_N(self,id):
        return self.G.nodes[id]['dPenalty']

    def getDCost_N(self,id):
        return self.G.nodes[id]['dCost']

    def getACost_N(self,id):
        return self.G.nodes[id]['aCost']

    def getActProb_N(self,id):
        return self.G.nodes[id]['actProb']

    def getTopoPosition_N(self,id):
        return self.G.nodes[id]['topoPosition']

    def getposActiveProb_N(self,id):
        return self.G.nodes[id]['posActiveProb']

    def getposInactiveProb_N(self,id):
        return self.G.nodes[id]['posInactiveProb']

    # Set Info

    def setState_N(self,id,value):
        try:
            if value != 0 and value != 1:
                raise Exception("Node state value must be 0 (Inactive) or 1 (Active).")
            else:
                self.G.nodes[id]['state'] = value
        except Exception as error:
            print(repr(error))

    def setType_N(self,id,value):
        try:
            if value != 0 and value != 1:
                raise Exception("Node type must be 0 (NONTARGET) or 1 (TARGET).")
            else:
                self.G.nodes[id]['type'] = value
        except Exception as error:
            print(repr(error))

    def setActivationType_N(self,id,value):
        try:
            if value != 0 and value != 1:
                raise Exception("Node eType must be 0 (OR) or 1 (AND).")
            else:
                self.G.nodes[id]['eType'] = value
        except Exception as error:
            print(repr(error))

    def setAReward_N(self,id,value):
        try:
            if value < 0:
                raise Exception("Node aReward must be greater than or equal to 0.")
            else:
                self.G.nodes[id]['aReward'] = value
        except Exception as error:
            print(repr(error))

    def setDPenalty_N(id,value):
        try:
            if value < 0:
                raise Exception("Node dPenalty must be greater than or equal to 0.")
            else:
                self.G.nodes[id]['dPenalty'] = value
        except Exception as error:
            print(repr(error))

    def setDCost_N(self,id,value):
        try:
            if value < 0:
                raise Exception("Node dCost must be greater than or equal to 0.")
            else:
                self.nodes[id]['dCost'] = value
        except Exception as error:
            print(repr(error))

    def setACost_N(self,id,value):
        try:
            if value < 0:
                raise Exception("Node aCost must be greater than or equal to 0.")
            else:
                self.nodes[id]['aCost'] = value
        except Exception as error:
            print(repr(error))

    def setActProb_N(self,id,value):
        try:
            if not self.isProb(value):
                raise Exception("Node action probability is not a valid probability (0>=<=1).")
            else:
                self.G.nodes[id]['actProb'] = value
        except Exception as error:
            print(repr(error))

    def setTopoPosition_N(self,id,value):
        ### ERROR CHECKING FOR LATER
        self.G.nodes[id]['topoPosition'] = value

    def setposActiveProb_N(self,id,value):
        try:
            if not self.isProb(value):
                raise Exception("Node posActive probability is not a valid probability (0>=<=1).")
            else:
                self.G.nodes[id]['posActiveProb'] = value
        except Exception as error:
            print(repr(error))

    def setposInactiveProb_N(self,id,value):
        try:
            if not self.isProb(value):
                raise Exception("Node posInctive probability is not a valid probability (0>=<=1).")
            else:
                self.G.nodes[id]['posInactiveProb'] = value
        except Exception as error:
            print(repr(error))

    # Edge Operations
    # Get Info
    def getid_E(self,edge):
        return self.G.edges[edge]['eid']

    def getType_E(self,edge):
        return self.G.edges[edge]['type']

    def getACost_E(self,edge):
        return self.G.edges[edge]['cost']

    def getActProb_E(self,edge):
        return self.G.edges[edge]['actProb']

    def getweight_E(self,edge):
        return self.G.edges[edge]['weight']

    # Set Info

    def setid_E(self, edge, value):
        self.G.edges[edge]['eid'] = value

    def setType_E(self,edge,value):
        try:
            if value != 0 and value != 1:
                raise Exception("Edge type must be either 0 (Normal) or 1 (Virtual).")
            else:
                 self.G.edges[edge]['type'] = value
        except Exception as error:
            print(repr(error))

    def setACost_E(self,edge,value):
        try:
            if value < 0:
                raise Exception("Cost for attacker on OR node must be greater than or equal to 0.")
            else:
                 self.G.edges[edge]['cost'] = value
        except Exception as error:
            print(repr(error))

    def setActProb_E(self,edge,value):
        try:
            if not self.isProb(value):
                raise Exception("Probability of successfully activating node through edge (For OR nodes only) is not a valid probability (0>=<=1).")
            else:
                self.G.edges[edge]['actProb'] = value
        except Exception as error:
            print(repr(error))

    def setweight_E(self,edge,value):
         self.G.edges[edge]['weight'] = value

    # Print Info
    def print_N(self,id):
        print(self.G.nodes[id])

    def print_E(self,edge):
        print(self.G.edges[edge])

    # Other Operations
    def getNumNodes(self):
        return self.G.number_of_nodes()

    def getNumEdges(self):
        return self.G.number_of_edges()

    def inDegree(self,id):
        return self.G.in_degree(id)

    def outDegree(self,id):
        return self.G.out_degree(id)

    def predecessors(self,id):
        return set(self.G.predecessors(id))

    def successors(self,id):
        return set(self.G.sucessors(id))

    def isDAG(self):
        return nx.is_directed_acyclic_graph(self.G)

    def getEdges(self):
        return self.G.edges()

    def get_ANDnodes(self):
        count = 0
        Andset = set()
        for node in G.nodes:
            if self.G.nodes[node]['eType'] == 1:
                count += 1
                Andset.add(node)
        return count, Andset

    def get_ORnodes(self):
        count = 0
        Orset = set()
        for node in G.nodes:
            if self.G.nodes[node]['eType'] == 0:
                count += 1
                Orset.add(node)
        return count, Orset

    def get_ORedges(self):
        ornodes = get_ORnodes(self.G)
        oredges = []
        for node in ornodes:
            oredges.append(self.G.in_edges(node))
        oredges = sortEdge(oredges)
        return oredges

    def get_Targets(self):
        count = 0
        targetset = set()
        for node in self.G.nodes:
            if self.G.nodes[node]['type'] == 1:
                count += 1
                targetset.add(node)
        return count,targetset

    def get_Roots(self):
        count = 0
        rootset = set()
        for node in self.G.nodes:
            if self.G.nodes[node]['root'] == 1:
                count += 1
                rootset.add(node)
        return count,rootset

    def get_num_NormalEdges(self):
        count = 0
        for edge in self.G.edges:
            if self.G.edges[edge]['type'] == 0:
                count += 1
        return count

    # Attributes Initialization
    def assignAttr_N(self,id,attr): #add code to check the lenth match
        self.G.nodes[id].update(dict(zip(G.nodes[id].keys(),attr)))

    def assignAttr_E(self,edge,attr):
        self.G.edges[edge].update(dict(zip(G.edges[edge].keys(),attr)))

	#def attrGenerator(num_nodes,num_edges,num_attr_N = 12,num_attr_E = 5,num_targets = 1,num_root = 1):
	    # Hard coding
	    #