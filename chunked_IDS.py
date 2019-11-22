# intrusion detection system using machine learning

import sys
import numpy as np
import sklearn as skl



###### Helper funcs
def chunkify_array(arr, chunksize):
    arr_len = len(arr)
    chunk_count = arr_len // chunksize
    last_chunk_index = arr_len - (arr_len % chunksize) 

    # split array into chunks
    chunks = np.split(arr, range(chunksize, arr_len, chunksize))    

    return chunks


def extract_arr_from_csv(filename):
    return np.genfromtxt(filename, delimiter=',', dtype=None, encoding='utf8')


def calc_chunked_survival_rate(chunk):
    # get all ids and their occurance
    msg_dict = dict()
    msg_count = len(chunk)

    for msg in chunk:
        ID = int(msg[1], 16)    #convert hex str to int
        if ID in msg_dict:
            msg_dict[ID] += 1
        else:
            msg_dict[ID] = 1

    # get probability of msg id
    for ID in msg_dict:
       msg_dict[ID] /= msg_count 

    #TODO: calc survival rate (from all chunks)

    # make list from msg_dict
    new_chunk = [ [k,v] for k,v in msg_dict.items() ]

    return new_chunk



##### test func
def test():
    gndfile = '../datasets/trainingData/HYUNDAI_SONATA_Attack_free_TRAIN_Release.csv'
    arr = extract_arr_from_csv(gndfile)

    chunksize = 100
    chunks = chunkify_array(arr, chunksize)

    chunk = calc_chunked_survival_rate(chunks[0])

    return chunk


#NOTE: maybe dont need to do all this if using ML?

###### Main func
def main():
    # get ground_truth from first arg
    if (len(sys.argv) == 1):
        test()
    else:
        ## read training data
        filename = sys.argv[1]
        raw_arr = extract_arr_from_csv(filename)

        ## calc survival rate 
        chunksize = 100     # article had best result for chunksize = 100
        chunks = chunkify_array(raw_arr, chunksize)
        
        ## make chunks back into normal array

        ## train machine

        ## detection
    return 




if __name__ == '__main__':
    main()
