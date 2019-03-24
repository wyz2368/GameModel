import os
import pickle

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

#This works for both dir and file.
def isExist(path):
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

# path = './gambit_data/'
# print(isExist(path))



