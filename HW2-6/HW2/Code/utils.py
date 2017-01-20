import utils
import numpy as np

def split_data(data_set):
    train_data = []
    train_target = []

    test_data = []
    test_target = []

    for i, d in enumerate(data_set.data):
        if i%7 == 0:
            test_data.append(d)
            test_target.append(data_set.target[i])
        else:
            train_data.append(d)
            train_target.append(data_set.target[i])
    return np.array(train_data), np.array(train_target), np.array(test_data), np.array(test_target)
