import json
import file_op as fp
import os

DIR_json = os.getcwd() + '/json_dir/'

def load_json_data(json_file):
    '''
    Loads the data from the file as Json into a new object.
    '''
    if not fp.isExist(DIR_json + json_file):
        raise ValueError(DIR_json + json_file + " does not exist.")
    with open(DIR_json + json_file) as data_file:
        result = json.load(data_file)
        return result

def save_json_data(file_name, json_obj):
    '''
    Prints the given Json object to the given file name.
    '''
    with open(DIR_json + file_name, 'w') as my_file:
        json.dump(json_obj, my_file)