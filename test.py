import numpy as np
import networkx as nx

G = nx.DiGraph()

G.add_edges_from([(1,2),(2,3),(3,4),(2,5),(5,6),(4,6)])

a = []

a += G.in_edges(2)

print(a)
print(G.number_of_nodes())
print(G.number_of_edges())
print(G.in_degree(6))
print(G.out_degree(2))
print(set(G.predecessors(6)))
print(set(G.successors(2)))
print(nx.is_directed_acyclic_graph(G))
print(G.edges())
print(G.nodes())
print(G.nodes[1])
print(G.in_edges(6))

a += G.in_edges(6)
print(a)