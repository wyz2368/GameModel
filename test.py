import numpy as np
import networkx as nx

a = [1,2,3,4,5,6]*2+[7,8,9,10,11,12]
history = 3
N = 6

if len(a)/N > history-1:
    a = a[(-history+1)*N:]

print(a)