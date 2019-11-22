# intrusion detection system using machine learning

import numpy as np



# reads csv from file and returns an array
def read_csv(filename):
    return np.genfromtxt(filename, delimiter=',', dtype=None, encoding='utf8')


# get the data and target 
def simple_extract_data(arr):
    arr_len = len(arr)
    data   = []
    target = []
    # iterate throgh each data point and extract relevent data and target
    for i in range(arr_len):
        # extract data from raw data
        can_id = int(arr[i][1], 16)
        data_point = [arr[i][0], can_id]
        # append datapoint and corresponding target to lists
        data.append(data_point) 
        target.append(arr[i][-1])

    return data, target



# gets the data and target from csv file
def data_from_csv(filename):
    arr = read_csv(filename)
    data, target = simple_extract_data(arr)
    return data, target
