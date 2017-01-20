import numpy as np
from sklearn.datasets import load_boston
from utils import split_data

import LinearRegression as LR 
import RidgeRegression as RR
import RidgeRegressionKCross as RK

def main(selected_features):

    data_set = load_boston()

    data_set.data = data_set.data[:, selected_features]
    train_data, train_target, test_data, test_target = split_data(data_set)

    lr = LR.LinearRegression()
    lr.fit(train_data, train_target)
    mse_test = lr.mse(test_data, test_target)
    mse_train = lr.mse(train_data, train_target)

    print "Sol. 3.3.a"
    print "Selected Features: {}, by (1-index): {}".format([data_set.feature_names[i] for i in selected_features], 
        [i + 1 for i in selected_features])

    print "\n"
    print "Linear Regression"
    print "{:^15}|{:^15}".format("Input Data", "MSE")
    print "{:^15}|{:^15.7}".format("test_data", mse_test)
    print "{:^15}|{:^15.7}".format("train_data", mse_train)
    #print "{:^15}|{:^15}".format()
    #print "mse_test: ", mse_test
    #print "mse_train: ", mse_train

    print "\n"
    print "Ridge Regression"
    print "{:^15}|{:^15}|{:^15}".format("Input Data", "Lambda", "MSE")
    print "-"*45
    ls = [0.01, 0.1, 1.0]
    for l in ls:
        rr = RR.RidgeRegression()
        W = rr.fit(train_data, train_target, l)
        
        mse_test = rr.mse(test_data, test_target)
        print "{:^15}|{:^15}|{:^15.7}".format("test_data", l, mse_test)
        mse_train = rr.mse(train_data, train_target)
        print "{:^15}|{:^15}|{:^15.7}".format("train_data", l, mse_train)
    
    print "\n"
    print "K-Fold Ridge Regression, K = 10"
    print "{:^15}|{:^15}".format("Lambda", "CVE")
    print "-"*45

    k = 10
    err = 0
    ls = [0.001, 0.01, 0.1, 1, 2, 3, 4,5 ,6 , 7, 8, 9, 10]
    
    res = []

    for l in ls:
        err = 0
        for i in range(k):
            k_set, k_set_target, rest_set, rest_set_target = RK.get_k_sets(train_data, train_target, k, offset=i)
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