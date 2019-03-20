import os
import pickle
import numpy as np

#TODO: path should include ./attackgraph

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if isExists:
        raise ValueError(path + " already exists.")
    else:
        os.makedirs(path)
        print(path + " has been created successfully.")

def rmdir(path):
    isExists = os.path.exists(path)
    if isExists:
        os.rmdir(path)
    else:
        raise ValueError(path + "does not exist.")

def rmfile(path):
    isExists = os.path.exists(path)
    if isExists:
        os.remove(path)
    else:
        raise ValueError(path + "does not exist.")

def isExist(path): #TODO: check if this means dir exists or file exists.
    return os.path.exists(path)

def isInName(str,name):
    return str in name


def save_pkl(obj,path):
    with open(path,'wb') as f:
        pickle.dump(obj,f)

def load_pkl(path):
    if not isExist(path):
        raise ValueError(path + " does not exist.")
    with open(path,'rb') as f:
        result = pickle.load(f)

    return result

# path = './attackgraph/payoff_matrix/a'

# a = np.ones((5,5))
#
# save_pkl(a,path)

# c = load_pkl(path)
# print(c)



