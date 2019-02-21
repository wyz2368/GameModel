import networkx as nx
import numpy as np
import sys
import random
import matplotlib.pyplot as plt

class Environment(object):

    def __init__(self, num_attr_N = 11, num_attr_E = 4, T=10, graphid=1, numNodes=20, numEdges=10, numRoot=3, numGoals=3, history = 3):
        self.num_attr_N = num_attr_N
        self.num_attr_E = num_attr_E
        self.T = T
        self.graphid = graphid
        self.G = nx.DiGraph(horizon = T, id = graphid)
        self.history = history

        # randomDAG parameters
        self.numNodes = numNodes
        self.numEdges = numEdges
        self.numRoot = numRoot
        self.numGoals = numGoals

    def daggenerator_wo_attrs(self,nodeset,edgeset):
        self.G.add_nodes_from(nodeset,
                         root = 0, # 0:NONROOT 1:ROOTNODE
                         type = 0, # 0:NONTARGET 1:TARGET
                         eType = 0,# 0:OR 1:AND node
                         state = 0,# 0:Inactive 1:Active
                         aReward = 0.0, # GREATER THAN OR EQUAL TO 0
                         dPenalty = 0.0, # SAME^
                         dCost = 0.0, # SAME^
                         aCost = 0.0, # SAME^
                         posActiveProb = 1.0, # prob of sending positive signal if node is active
                         posInactiveProb = 0.0, # prob of sending positive signal if node is inactive(false alarm)
                         actProb = 1.0) # prob of becoming active if being activated, for AND node only
        self.G.add_edges_from(edgeset,
                         eid = -1,
                         type = 0, # 0:NORMAL 1:VIRTUAL
                         cost = 0, # Cost for attacker on OR node, GREATER THAN OR EQUAL TO 0
                         actProb=1.0) # probability of successfully activating, for OR node only

    def randomDAG(self, NmaxAReward=100, NmaxDPenalty=100, NmaxDCost=100, NmaxACost=100, EmaxACost=100, EminWeight=0, EmaxWeight=100):
        # Exception handling
        # try:
        #     if self.numRoot + self.numGoals > self.numNodes:
        #         raise Exception("(Number of root nodes) + (Number of goal nodes) cannot exceed total number of nodes.")
        # except Exception as error:
        #     print(repr(error))
        #     return 1
        try:
            maxEdges = (self.numNodes-1)*(self.numNodes)/2
            if self.numEdges > maxEdges:
                raise Exception("For a graph with " + str(self.numNodes) + " nodes, there can be a maximum of " + str(int(maxEdges)) + " edges.")
        except Exception as error:
            print(repr(error))
            return 1

        self.G = nx.gnp_random_graph(self.numNodes, 1, directed=True) # Create fully connected directed Erdos-Renyi graph.
        self.G = nx.DiGraph([(u,v) for (u,v) in self.G.edges() if u<v], horizon = self.T, id = self.graphid) # Drop all edges (u,v) where edge u<v to enforce acyclic graph property.
        rootNodes = random.sample(range(1,self.numNodes-1),self.numRoot-1) # Given the parameter self.numRoot, pick self.numRoot-1 random root IDs.
                                                                 # Node 0 will also always be root. Last node (ID:self.numNodes) cannot be root node.
        goalNodes = random.sample(list(set(range(1,self.numNodes))-set(rootNodes)),self.numGoals) # Randomly pick GoalNodes

        for rootNode in rootNodes: # Out of the picked rootNodes, drop all edges (u,v) where v = rootNode.
            for start in range(0, rootNode):
                self.G.remove_edge(start, rootNode)
        canRemove = list(self.G.edges)
        while len(self.G.edges) > self.numEdges and len(canRemove) != 0: # Randomly delete edges until self.numEdges is met, or if there are no more nodes to remove.
                                                                    # canRemove = nodes not yet removed OR nodes that once removed do not
                                                                    # break the connected property.
            deleteEdge = random.choice(canRemove)
            self.G.remove_edge(deleteEdge[0], deleteEdge[1])
            if (not nx.is_connected(self.G.to_undirected())) or (len(self.G.pred[deleteEdge[1]]) == 0):
                self.G.add_edge(deleteEdge[0], deleteEdge[1])
            canRemove.remove(deleteEdge)

        # Set random node attributes
        for nodeID in range(self.numNodes):
            if len(self.G.pred[nodeID]) == 0:
                self.setRoot_N(nodeID, 1)
                self.setType_N(nodeID, 0) # Root nodes cannot be target (goal) nodes.
            else:
                self.setRoot_N(nodeID, 0)
                if nodeID in goalNodes: # Set Goal nodes
                    self.setType_N(nodeID, 1)
                else:
                    self.setType_N(nodeID, 0)
            self.setActivationType_N(nodeID, np.random.randint(2))
            self.setState_N(nodeID, np.random.randint(2))
            self.setAReward_N(nodeID, np.random.uniform(0, NmaxAReward))
            self.setDPenalty_N(nodeID, np.random.uniform(0, NmaxDPenalty))
            self.setDCost_N(nodeID, np.random.uniform(0, NmaxDCost))
            self.setACost_N(nodeID, np.random.uniform(0, NmaxACost))
            self.setposActiveProb_N(nodeID, np.random.uniform(0, 1))
            self.setposInactiveProb_N(nodeID, np.random.uniform(0, 1))
            self.setTopoPosition_N(nodeID, -1)

        # Set random edge attributes
        for edgeID, edge in enumerate(self.G.edges):
            self.setid_E(edge, edgeID)
            self.setType_E(edge, np.random.randint(2))
            self.setACost_E(edge, np.random.uniform(0, EmaxACost))
            self.setweight_E(edge, np.random.uniform(EminWeight, EmaxWeight))
            self.setActProb_E(edge, np.random.uniform(0, 1))

       # Parameter Format:
       #    AttributesDict: Dictionary of the following attributes:
       #	Nodes = List of N integers, representing Node IDs.
       #    Edges = List of E Tuples, representing Edges.
       #    Nroots, Ntypes, NeTypes, Nstates, NaRewards, NdPenalties, NdCosts, NaCosts, NposActiveProbs, NposInacriveProbs, NtopoPositions: 
       #    Size N list. Each List[x] attribute correspondes to the node in position nodes[x].
    def specifiedDAG(self, attributesDict):
        self.daggenerator_wo_attrs(nodes,edges,T,graphid)
        for nodeID in range(attributesDict[nodes]):
            self.setRoot_N(nodeID, attributesDict[Nroots[nodeID]])
            self.setType_N(nodeID, attributesDict[Ntypes[nodeID]])
            self.setActivationType_N(nodeID, attributesDict[NeTypes[nodeID]])
            self.setState_N(nodeID, attributesDict[Nstates[nodeID]]) # Are all nodes inactive at the start? Or not?
            self.setAReward_N(nodeID, attributesDict[NaRewards[nodeID]])
            self.setDPenalty_N(nodeID, attributesDict[NdPenalties[nodeID]])
            self.setDCost_N(nodeID, attributesDict[NdCosts[nodeID]])
            self.setACost_N(nodeID, attributesDict[NaCosts[nodeID]])
            self.setposActiveProb_N(nodeID, attributesDict[NposActiveProbs[nodeID]])
            self.setposInactiveProb_N(nodeID, attributesDict[NposInactiveProbs[nodeID]])
            self.setTopoPosition_N(nodeID, attributesDict[NtopoPositions[nodeID]])
        for edge in range(attributesDict[edges]):
            self.setid_E((attributesDict[edges[0]], attributesDict[edges[1]]), attributesDict[Eeids[edge]])
            self.setType_E((attributesDict[edges[0]], attributesDict[edges[1]]), attributesDict[Etypes[edge]])
            self.setACost_E((attributesDict[edges[0]], attributesDict[edges[1]]), attributesDict[Ecosts[edge]])
            self.setweight_E((attributesDict[edges[0]], attributesDict[edges[1]]), attributesDict[Eweights[edge]])
            self.setActProb_E((attributesDict[edges[0]], attributesDict[edges[1]]), attributesDict[actProb[edge]])

    # Visualizes DAG
    # Node did not visualize: aReward, dPenalty, dCost, aCost, posActiveProb, posInactiveProb, actProb, topoPosition
    # Edge did not visualize: eid, cost, weight, actProb
    def visualize(self):
        nodePos = nx.layout.spring_layout(self.G)
        # Local variable initialization
        try: # rootNodes and targetNodes cannot overlap
            rootNodes = self.get_Roots()[1]
            targetNodes = self.get_Targets()[1]
            if bool(set(rootNodes) & set(targetNodes)):
                raise Exception("Goal and Root nodes overlap. A Goal node cannot be a Root node, and vice versa.")
        except Exception as error:
            print(repr(error))
            return 1
        virtualEdges = [edge for edge in self.G.edges if self.getType_E(edge) == 1]

        # Visualization format: Nodes
        #    Active = Green, Inactive = Red
        #    nonGoal AND Node = ^ Triangle
        #    Goal AND Node = p Pentagon
        #    nonGoal OR Node = o Circle
        #    Goal OR Node = h Hexagon
        #	 ROOT nodes = Bold Labels
        #	 nonROOT nodes = Regular Labels
        nodeSize = 300
        for node in self.G.nodes:
            if self.getState_N(node) == 1:
                nodeColor = 'g' # Active = Green
            else:
                nodeColor = 'r' # Inactive = Red
            if self.getActivationType_N(node) == 1:
                if node in targetNodes:
                    nodeShape = 'p' # Goal AND Node = p Pentagon
                else:
                    nodeShape = '^' # nonGoal AND Node = ^ Triangle
            else:
                if node in targetNodes:
                    nodeShape = 'h' # Goal OR Node = h Hexagon
                else:
                    nodeShape = 'o' # nonGoal OR Node = o Circle
            nx.draw_networkx_nodes(self.G, nodePos, node_shape=nodeShape, nodelist=[node], node_size=nodeSize, node_color=nodeColor, vmax=0.1)
        nx.draw_networkx_labels(self.G, nodePos, labels={k: k for k in rootNodes}, font_weight='bold') # ROOT nodes = Bold Labels
        nx.draw_networkx_labels(self.G, nodePos, labels={k: k for k in list(set(self.G.nodes)-set(rootNodes))}) # nonROOT nodes = Regular Labels

        # Visualization format: Edges
        # 	Virtual edges = Blue
        # 	Normal edges = Black
        nx.draw_networkx_edges(self.G, nodePos, edgelist=virtualEdges, edge_color='blue') # Virtual edges = Blue
        nx.draw_networkx_edges(self.G, nodePos, edgelist=list(set(self.G.edges)-set(virtualEdges))) # Normal edges = Black

        plt.show()

    def isProb(self,p):
        return p >= 0.0 and p <= 1.0

    def sortEdge(self,edgeset):
        sorted_by_first_second = sorted(edgeset, key=lambda tup: (tup[0], tup[1]))
        return sorted_by_first_second

    # Graph Operation
    def getHorizon_G(self):
        return self.G.graph['horizon']

    def setHorizon_G(self,value):
        self.T = value
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

    def getposActiveProb_N(self,id):
        return self.G.nodes[id]['posActiveProb']

    def getposInactiveProb_N(self,id):
        return self.G.nodes[id]['posInactiveProb']

    # Set Info

    def setRoot_N(self, id, value):
        try:
            if value != 0 and value != 1:
                raise Exception("Node root value must be 0 (NONROOT) or 1 (ROOT).")
            else:
                self.G.nodes[id]['root'] = value
        except Exception as error:
            print(repr(error))

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

    def setDPenalty_N(self,id,value):
        try:
            if value > 0:
                raise Exception("Node dPenalty must be less than or equal to 0.")
            else:
                self.G.nodes[id]['dPenalty'] = value
        except Exception as error:
            print(repr(error))

    def setDCost_N(self,id,value):
        try:
            if value > 0:
                raise Exception("Node dCost must be less than or equal to 0.")
            else:
                self.G.nodes[id]['dCost'] = value
        except Exception as error:
            print(repr(error))

    def setACost_N(self,id,value):
        try:
            if value > 0:
                raise Exception("Node aCost must be less than or equal to 0.")
            else:
                self.G.nodes[id]['aCost'] = value
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
        return set(self.G.successors(id))

    def isDAG(self):
        return nx.is_directed_acyclic_graph(self.G)

    def getEdges(self):
        return self.G.edges()

    def get_ANDnodes(self):
        count = 0
        Andset = set()
        for node in self.G.nodes:
            if self.G.nodes[node]['eType'] == 1:
                count += 1
                Andset.add(node)
        return count, Andset

    def get_ORnodes(self):
        count = 0
        Orset = set()
        for node in self.G.nodes:
            if self.G.nodes[node]['eType'] == 0:
                count += 1
                Orset.add(node)
        return count, Orset

    def get_ORedges(self):
        ornodes = self.get_ORnodes()
        oredges = []
        for node in ornodes:
            oredges.append(self.G.in_edges(node))
        oredges = self.sortEdge(oredges)
        return len(oredges), oredges

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

    def get_NormalEdges(self):
        count = 0
        normaledge = set()
        for edge in self.G.edges:
            if self.G.edges[edge]['type'] == 0:
                count += 1
                normaledge.add(edge)
        return count, self.sortEdge(normaledge)

    # Attributes Initialization
    def assignAttr_N(self,id,attr): #add code to check the lenth match
        self.G.nodes[id].update(dict(zip(self.G.nodes[id].keys(),attr)))

    def assignAttr_E(self,edge,attr):
        self.G.edges[edge].update(dict(zip(self.G.edges[edge].keys(),attr)))

    #def attrGenerator(num_nodes,num_edges,num_attr_N = 12,num_attr_E = 5,num_targets = 1,num_root = 1):
    # Hard coding
    #

    ### API FUNCTIONS ###

    # attact and defact are attack set and defence set
    def _step(self,attact,defact):
        # immediate reward for both players
        aReward = 0
        dReward = 0
        # T = self.G.graph['horizon'] # if discounted, should count how many times _step is called.
        #attacker's action
        for attack in attact:
            if isinstance(attack,tuple):
                #check OR node
                aReward += self.G.edges[attack]['cost']
                if random.uniform(0,1) <= self.G.edges[attack]['actProb']:
                    self.G.nodes[attack[-1]]['state'] = 1
            else:
                #check AND node
                aReward += self.G.nodes[attack]['aCost']
                if random.uniform(0,1) <= self.G.nodes[attack]['actProb']:
                    self.G.nodes[attack]['state'] = 1
        #defender's action
        for node in defact:
            self.G.nodes[node]['state'] = 0
            dReward += self.G.nodes[node]['dCost']
        _,targetset = self.get_Targets()
        for node in targetset:
            if self.G.nodes[node]['state'] == 1:
                aReward += self.G.nodes[node]['aReward']
                dReward += self.G.nodes[node]['dPenalty']
        #if goal node prevails for next time step
        # return true state and obs
        return self.get_att_isActive(),self.get_def_hadAlert(),aReward,dReward

    #return a list of indicator of whether node is activated.
    def get_att_isActive(self):
        isActive = []
        for id in np.arange(self.getNumNodes()):
            if self.G.nodes[id+1]['state'] == 1:
                isActive.append(1)
            else:
                isActive.append(0)
        return isActive

    # can be called only once for each time step.
    def get_def_hadAlert(self):
        alert = []
        for node in self.G.nodes:
            if self.G.nodes[node]['state'] == 1:
                if random.uniform(0, 1) <= self.G.nodes[node]['posActiveProb']:
                    alert.append(1)
                else:
                    alert.append(0)
            elif self.G.nodes[node]['state'] == 0:
                if random.uniform(0, 1) <= self.G.nodes[node]['posInactiveProb']:
                    alert.append(1)
                else:
                    alert.append(0)
            else:
                raise ValueError("node state is abnormal.")
        return alert

    #reset the environment, G_reserved is a copy of the initial env
    def save_graph_copy(self):
        self.G_reserved = self.G.copy()


    def reset(self):
        self.G = self.G_reserved.copy()

    #other APIs similar to OpenAI gym
    def obs_dim_att(self):
        num_andnode, _ = self.get_ANDnodes()
        num_oredges, _ = self.get_ORedges()
        return self.G.number_of_nodes() + 2*(num_andnode + num_oredges) + 1

    def obs_dim_def(self):
        N = self.G.number_of_nodes()
        return self.history*N*2 + N + 1 # NO CNN

    def act_dim_att(self):
        num_andnode, _ = self.get_ANDnodes()
        num_oredges, _ = self.get_ORedges()
        return num_andnode + num_oredges + 1 #pass

    def act_dim_att(self):
        return self.G.number_of_nodes() + 1 #pass



"""
test = Environment()
test.randomDAG(1,1)
test.visualize()
"""
