# intrusion detection system using machine learning

import numpy as np
import csv



# reads csv from file and returns an array
def read_csv(filename):
    return np.genfromtxt(filename, delimiter=',', dtype=None, encoding='utf8')

def write_data_to_csv(filename, data):
    with open(filename, 'w') as csvfile:
        csv.writer(filename)
        for msgno in range(len(data)):
            csv.writerow([msgno+1, msg[msgno]])
    return
    

# get the data and target 
def simple_extract_data(arr):
    arr_len = len(arr)
    data   = []
    target = []
    # iterate throgh each data point and extract relevent data and target
    start_time = arr[0][0]
    for i in range(arr_len):
        # extract data from raw data
        can_id = int(arr[i][1], 16)
        can_timestamp = arr[i][0] - start_time
        data_point = [can_timestamp, can_id]
        # append datapoint and corresponding target to lists
        data.append(data_point) 
        target.append(arr[i][-1])

    return data, target



# get the data and target 
def simple_extract_test_data(arr):
    arr_len = len(arr)
    data   = []
    start_time = arr[0][1]
    # iterate throgh each data point and extract relevent data and target
    for i in range(arr_len):
        # extract data from raw data
        can_id = int(arr[i][2], 16)
        can_timestamp = arr[i][1] - start_time
        data_point = [can_timestamp, can_id]
        # append datapoint and corresponding target to lists
        data.append(data_point) 

    return data




# gets the data and target from csv file
def data_from_csv(filename):
    arr = read_csv(filename)
    data, target = simple_extract_data(arr)
    return data, target


def data_from_test_csv(filename):
    arr = read_csv(filename)
    data = simple_extract_test_data(arr)
    return data
