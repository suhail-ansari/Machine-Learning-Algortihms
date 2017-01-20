import numpy as np
from sklearn.datasets import load_boston
from scipy.stats import mstats

from utils import split_data

class RidgeRegression(object):
    def __init__(self):
        pass
    
    def fit(self, train_data, train_target, l):
        self.means = np.mean(train_data, axis=0)
        self.std_devs = np.std(train_data, axis=0)

        self.normalized_data = np.ones(shape=(train_data.shape[0], train_data.shape[1] + 1))
        self.normalized_data[:, 1:] = mstats.zscore(train_data, axis=0)

        X = self.normalized_data
        Y = train_target
        self.W = np.linalg.pinv(X.T.dot(X) - (l*np.eye(X.shape[1], X.shape[1]))).dot(X.T).dot(Y)
    
    def predict(self, test_data):
        temp_data = np.ones(shape=(test_data.shape[0], test_data.shape[1] + 1))
        for i, column in enumerate(test_data.T):
            temp_data[:, i + 1] =  (column - self.means[i])/self.std_devs[i]
        return temp_data.dot(self.W)
    
    def mse(self, test_data, test_target):
        p = self.predict(test_data)
        loss = p - test_target
        return (np.square(loss).sum())/test_data.shape[0]

def main():
    data_set = load_boston()

    train_data, train_target, test_data, test_target = split_data(data_set)

    print "\n"
    print "Ridge Regression"
    print "{:^15}|{:^15}|{:^15}".format("Input Data", "Lambda", "MSE")
    print "-"*45
    ls = [0.01, 0.1, 1.0]
    for l in ls:
        rr = RidgeRegression()
        rr.fit(train_data, train_target, l)
        mse_train = rr.mse(train_data, train_target)
        mse_test = rr.mse(test_data, test_target)

        print "{:^15}|{:^15}|{:^15.7}".format("test_data", l, mse_test)
        print "{:^15}|{:^15}|{:^15.7}".format("train_data", l, mse_train)

if __name__ == "__main__":
    main()
    





