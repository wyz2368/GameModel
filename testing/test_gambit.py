import gambit_analysis as ga
import numpy as np
import file_op as fp
import time

# poDef = np.array([[3,0],[1,2]])
# poAtt = np.array([[2,0],[0,3]])
#
# ga.encode_gambit_file(poDef,poAtt)
#
# ga.gambit_analysis()

# ga.decode_gambit_file()

n = 100
poDef = np.random.normal(size=(n,n))
poAtt = np.random.normal(size=(n,n))
poDef = np.round(poDef,2)
poAtt = np.round(poAtt,2)

# print(poAtt)

t1 = time.time()
nash_att, nash_def = ga.do_gambit_analysis(poDef, poAtt)
t2 = time.time()

print("time:",t2-t1)
print(nash_att, nash_def)

# nash_att, nash_def = ga.decode_gambit_file()
# print(nash_att, nash_def)

# a = '19/30,0,11/30,0,0,0,0,0,0,0,34/101,0,0,0,67/101,0,0,0,0,0'
# b = a.split(',')
# b = float(b)
# c = np.array(b,dtype=np.float)
# print(c)
# print(c[0])


