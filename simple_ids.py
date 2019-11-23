# intrusion detection system using machine learning

import sys
import numpy as np
import sklearn as skl
from sklearn import svm
from CAN_import import data_from_csv


# globals
attack = 'T'
normal = 'R'

# ids class
class ids:

    # attributes
    clfs = []   # holds all classifiers

    # init ids class and teach the machine learning algorithms
    def __init__(self, msg_files):
        for msg_file in msg_files:
            # init clasifier for file
            clf = svm.SVC(gamma=0.001, C=100.)
            
            # get data from relevent file
            msgs, msg_type = data_from_csv(msg_file)
            msgs_len = len(msgs)

            # get split index 
            splitpoint = msgs_len - 2000

            # fit msgs up to splitpoint, leaving the rest to check accuracy
            clf.fit(msgs[:splitpoint], msg_type[:splitpoint])
            self.clfs.append(clf)


    # predict the type of message with Machine Learning.
    # this willl be subject to high levels of false +ves
    def predict_msg_type(self, msgs):
        assert len(self.clfs) > 1, 'No classifiers stored in object.'
        msg_types = [normal] * len(msgs)    # make default to normal
        # check with each classifier for attacks
        for clf in self.clfs:
            preds = clf.predict(msgs)
            # update predictions
            for i in range(len(preds)):
                if preds[i] == attack:
                    msg_types[i] = attack

        return msg_types 
        





##### helper funcs
def get_accuracy(target, pred):
    pred_len = len(pred)
    accuracy = 0
    # count how many were correct
    for i in range(pred_len):
        if target[i] == pred[i]:
            accuracy += 1
    return accuracy/pred_len


def clfs_from_files(files):
    # vars
    clfs  = []
    preds = []

    # make classifiers from csv files
    for i in range(len(files)):
        # init clasifier for file
        clf = svm.SVC(gamma=0.001, C=100.)
        
        # get data from relevent file
        data, target = data_from_csv(files[i])
        data_len = len(data)

        # get split index 
        splitpoint = data_len - 2000

        # fit data up to splitpoint, leaving the rest to check accuracy
        clf.fit(data[:splitpoint], target[:splitpoint])
        print('data fitting for file[{}] complete'.format(i))

        # check accuracy of output
        pred = clf.predict(data[splitpoint:])
        accuracy = get_accuracy(target[splitpoint:], pred)
        print('accuracy is {}'.format(accuracy))
        
        # store clf and pred
        preds.append(pred)
        clfs.append(clf)
    return clfs, preds


    

##### main func
def main():
    # set file names
    data_dir = '../datasets/trainingData/'
    car_name = 'HYUNDAI_SONATA'
    file_nml = data_dir + car_name + '_Attack_free_TRAIN_Release.csv'
    file_fld = data_dir + car_name + '_Flooding_TRAIN_Release.csv'
    file_fuz = data_dir + car_name + '_Fuzzy_TRAIN_Release.csv'
    file_mal = data_dir + car_name + '_Malfunction_TRAIN_Release.csv'
    files = [file_fld, file_fuz, file_mal]
    
    # make classifiers and get predictions for all files
    myIDS = ids(files)
    msg, msg_type  = data_from_csv(file_mal)
    splitpoint = len(msg) - 2000

    pred = myIDS.predict_msg_type(msg[splitpoint:])
    accuracy = get_accuracy(msg_type[splitpoint:], pred)
    
    print('accuracy is {}'.format(accuracy))

    return myIDS




if __name__ == '__main__':
    main()
