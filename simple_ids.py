# intrusion detection system using machine learning

import sys
import numpy as np
import sklearn as skl
from sklearn import svm
from CAN_import import extract_arr_from_csv



##### test func
def main():
    data_dir     = '../datasets/trainingData/'
    file_normal  = data_dir + 'HYUNDAI_SONATA_Attack_free_TRAIN_Release.csv'
    file_flood   = data_dir + 'HYUNDAI_SONATA_Flooding_TRAIN_Release.csv'
    file_fuzzy   = data_dir + 'HYUNDAI_SONATA_Fuzzy_TRAIN_Release.csv'
    file_malfunc = data_dir + 'HYUNDAI_SONATA_Malfunction_TRAIN_Release.csv'

    arr     = extract_arr_from_csv(file_flood)
    arr_len = len(arr)

    # split data
    data   = [ [arr[i][0], int(arr[i][1], 16)] for i in range(arr_len) ]
    target = [ arr[i][-1] for i in range(arr_len) ]

    clf = svm.SVC(gamma=0.001, C=100.)

    splitpoint = arr_len - 10

    clf.fit(data[:splitpoint], target[:splitpoint])
    print('data fitting complete')

    pred = clf.predict(data[splitpoint:])
    print("prediction of last msg = {}".format(pred))
    return clf




if __name__ == '__main__':
    main()
