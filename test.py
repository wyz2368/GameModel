import networkx as nx
import random
import numpy as np

G = nx.DiGraph()
G.add_nodes_from([1,2,3],time=2,day=3,hour=24)
G.add_edges_from([(1,2),(1,3)],a1=1,a2=2,a3=3)
print(G.in_edges(3))

H = G.copy()
H.nodes[1]['sadf'] = 1
print(H.nodes.data())
print(G.nodes.data())
