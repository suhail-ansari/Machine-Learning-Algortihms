import numpy as np
from scipy.stats import mstats
from sklearn.datasets import load_boston
import RidgeRegression as RR
from utils import split_data

def get_k_sets(train_data, target_data, k, offset=0):
    n = train_data.shape[0]
    s = int(n/k)
    k_set_data = train_data[(k*offset):(k*offset + s)]
    k_set_target = target_data[(k*offset):(k*offset + s)]

    if offset > 0:
        rest_data = train_data.copy()
        rest_data_target = target_data.copy()
        rest_data = np.delete(rest_data, range((s*offset), (s*offset + s + 1)), axis = 0)
        rest_data_target = np.delete(rest_data_target, range((s*offset), (s*offset + s + 1)), axis=0)
    else:
        rest_data = train_data[(s*offset + s + 1):]
        rest_data_target = target_data[(s*offset + s + 1):]

    return k_set_data, k_set_target.reshape((k_set_target.shape[0], 1)), rest_data, rest_data_target.reshape(rest_data_target.shape[0], 1)

def main():
    data_set = load_boston()
    train_data, train_target, test_data, test_target = split_data(data_set)

    print "\n"
    print "K-Fold Ridge Regression, K = 10"
    print "{:^15}|{:^15}".format("Lambda", "CVE")
    print "-"*30

    k = 10
    err = 0
    ls = [0.001, 0.01, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    res = []

    for l in ls:
        err = 0
        for i in range(k):
            k_set, k_set_target, rest_set, rest_set_target = get_k_sets(train_data, train_target, k, offset=i)
            rr = RR.RidgeRegression()
            rr.fit(rest_set, rest_set_target, l)
            err += rr.mse(k_set, k_set_target)
        res.append([l, err/k])
        print "{:^15}|{:^15.7}".format(l, (err/k))
    

    min_l = min(res, key=lambda z: z[1])[0]
    
    print "\n"
    print "Select L = {} with min CVE".format(min_l)
    
    rr = RR.RidgeRegression()
    rr.fit(train_data, train_target, min_l)    
    mse_test = rr.mse(test_data, test_target)
    mse_train = rr.mse(train_data, train_target)

    print "{:^15}|{:^15}|{:^15}".format("Input Data", "Lambda", "MSE")
    print "-"*45
    print "{:^15}|{:^15}|{:^15.7}".format("test_data", min_l, mse_test)
    print "{:^15}|{:^15}|{:^15.7}".format("train_data", min_l, mse_train)
    print "\n"

if __name__ == "__main__":
    main()
    