import numpy as np

from scipy.stats import mstats
from sklearn.datasets import load_boston

from utils import split_data


class LinearRegression(object):
    def __init__(self):
        pass
    
    def fit(self, train_data, train_target):
        self.means = np.mean(train_data, axis=0)
        self.std_devs = np.std(train_data, axis=0)

        self.normalized_data = np.ones(shape=(train_data.shape[0], train_data.shape[1] + 1))
        self.normalized_data[:, 1:] = mstats.zscore(train_data, axis=0)

        X = self.normalized_data
        Y = train_target
        self.W = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(Y)
        
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

    lr = LinearRegression()
    lr.fit(train_data, train_target)
    mse_train = lr.mse(train_data, train_target)
    mse_test = lr.mse(test_data, test_target)

    print "\n"
    print "Linear Regression"
    print "{:^15}|{:^15}".format("Input Data", "MSE")
    print "-"*30
    print "{:^15}|{:^15.7}".format("test_data", mse_test)
    print "{:^15}|{:^15.7}".format("train_data", mse_train)
    print "\n"

if __name__ == "__main__":
    main()
    



