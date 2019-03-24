import game_data
import DagGenerator as dag
import numpy as np
import random
import time
import util

random.seed(2)
np.random.seed(2)
t1 = time.time()
env = dag.Environment(numNodes=5, numEdges=4, numRoot=2, numGoals=1)
env.randomDAG()
t2 = time.time()

game = game_data.Game_data(env, 4, 2,[256, 256], 400)

# env.G.nodes[1]['root'] = 0
#
# print(env.G.nodes.data())
# print(game.env.G.nodes.data())
# print(env.G.edges)
# print(game.env.G.edges)


game.init_payoffmatrix(5,-2)
print(game.payoffmatrix_att)
print(game.payoffmatrix_def)

ne = {}
ne[0] = np.array([0.4,0.6])
ne[1] = np.array([0.4,0.6])

print(ne)
game.add_nasheq(2,ne)
# print(game.nasheq)

# game.add_col_att(np.array([[1]]))
# game.add_col_def(np.array([[2]]))
# game.add_row_att(np.array([[1,2]]))
# game.add_row_def(np.array([[2,3]]))

# print(np.shape(game.payoffmatrix_att))
# print(np.shape(np.array([[1,2]])))

# print(game.payoffmatrix_att)
# print(game.payoffmatrix_def)

game.payoffmatrix_att = np.array([[1,1],[1,1]])
game.payoffmatrix_def = np.array([[1,5],[6,1]])

aPayoff, dPayoff = util.payoff_mixed_NE(game,2)
print(aPayoff, dPayoff)

















