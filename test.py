import networkx as nx
import numpy as np

G = nx.DiGraph()
G.add_nodes_from([1,2,3],time=2,day=3,hour=24)
G.add_edges_from([(1,2),(1,3)],a1=1,a2=2,a3=3)
print(G.nodes[1].keys())
print(G.edges[(1,2)])
print(set(G.successors(1)))
print(nx.is_directed_acyclic_graph(G))

# d=[1,2,3]
# dict={}
# dict=dict.fromkeys(d,)
# print(dict)

A = np.random.random((3,3))

a = A[1,:]
b = G.edges[(1,2)].keys()
G.edges[(1,2)].update(dict(zip(b,a)))
print(G.edges[1,2])

