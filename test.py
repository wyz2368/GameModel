import numpy as np
import math
import os
# import networkx as nx
# import random
# import itertools
# import time
# import training
# import tensorflow as tf
import pickle as pk

# G = nx.DiGraph()
#
# G.add_edges_from([(1,2),(2,3),(3,4),(2,5),(5,6),(4,6)])
#
# a = []
#
# a += G.in_edges(2)

# print(a)
# print(G.number_of_nodes())
# print(G.number_of_edges())
# print(G.in_degree(6))
# print(G.out_degree(2))
# print(set(G.predecessors(6)))
# print(set(G.successors(2)))
# print(nx.is_directed_acyclic_graph(G))
# print(G.edges())
# print(list(G.nodes()))
# print(G.nodes[1])
# print(G.in_edges(6))
#
# a += G.in_edges(6)
# print(a)

# G1 = nx.DiGraph()
# G1.add_node(3)
# G1.add_node(1)
# G1.add_node(0)
# G1.add_node(7)
# G1.add_node(2)
# G1.add_node(11)
# print(sorted(G1.nodes))

# t1 = time.time()
# a = []
# for i in range(400):
#     a.append(G.copy())
# t2 = time.time()
# print(t2-t1)
# print(a[0].nodes)
# a[1].add_node(10)
# print(a[1].nodes)

# random_actions = tf.random_uniform([10], minval=0, maxval=5, dtype=tf.int64)
# sess = tf.Session()
# print(sess.run(random_actions))


# print(isinstance(os.getcwd() + '/defender_strategies/',str))

# a = ['1','2','3']
# print(np.random.choice(a, p=np.array([0.3,0.3,0.4])))


# class Dog(object):
#     def top(self,env):
#         self.mytop = env
#
# class Animal(object):
#     def __init__(self):
#         self.dog = Dog()
#         self.age = 5
#
#
# a = Animal()
#
# a.dog.top(a)
#
# print(a.dog.mytop.age)

# new_dim = 6
# position_col_list = []
# position_row_list = []
# for i in range(new_dim - 1):
#     position_col_list.append((i, new_dim - 1))
# for j in range(new_dim):
#     position_row_list.append((new_dim - 1, j))
#
# print(position_col_list)
# print(position_row_list)


from baselines.deepq.deepq import learn

























