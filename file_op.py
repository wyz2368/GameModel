import os
import json

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

def get_json_data(json_file):
    '''
    Loads the data from the file as Json into a new object.
    '''
    with open(json_file) as data_file:
        result = json.load(data_file)
        return result

def print_json(file_name, json_obj):
    '''
    Prints the given Json object to the given file name.
    '''
    with open(file_name, 'w') as my_file:
        json.dump(json_obj, my_file)



