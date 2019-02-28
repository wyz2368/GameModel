import numpy as np
import networkx as nx

G = nx.DiGraph()
G.add_node(1)
print(G.in_degree(1))