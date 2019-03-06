import os

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

def isExist(path):
    return os.path.exists(path)

def isInName(str,name):
    return str in name




