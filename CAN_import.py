# intrusion detection system using machine learning

import numpy as np

def extract_arr_from_csv(filename):
    return np.genfromtxt(filename, delimiter=',', dtype=None, encoding='utf8')

