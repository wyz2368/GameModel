import DagGenerator as dag
import numpy as np

env = dag.Environment(numNodes=5, numEdges=4, numRoot=2, numGoals=1)
env.randomDAG()
