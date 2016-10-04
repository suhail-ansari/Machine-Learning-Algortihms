import numpy as np
from scipy.stats import mstats

class RidgeRegression(object):
    def __init__(self):
        """
        Create Ridge Regressor Instance
        """
        pass
    
    def fit(self, train_data, train_target, l):
        """
        fit training data

        Parameters:
            X: Training Data
            Y: Target Values
            l: Lambda
        """
        self.means = np.mean(train_data, axis=0)
        self.std_devs = np.std(train_data, axis=0)

        self.normalized_data = np.ones(shape=(train_data.shape[0], train_data.shape[1] + 1))
        self.normalized_data[:, 1:] = mstats.zscore(train_data, axis=0)

        X = self.normalized_data
        Y = train_target
        self.W = np.linalg.pinv(X.T.dot(X) - (l*np.eye(X.shape[1], X.shape[1]))).dot(X.T).dot(Y)
    
    def predict(self, test_data):
        """
        predict values from sample data
        
        Parameters:
            X: Testing Data
        
        Returns:
            P: Predicted Values
        """
        temp_data = np.ones(shape=(test_data.shape[0], test_data.shape[1] + 1))
        for i, column in enumerate(test_data.T):
            temp_data[:, i + 1] =  (column - self.means[i])/self.std_devs[i]
        return temp_data.dot(self.W)
    
    def mse(self, test_data, test_target):
        """
        Mean Square Error of predicted values and actual target values  
        
        Parameters:
            X: Testing Data
            Y: Actual Target Values
        
        Returns:
            E: Mean Square Error
        """
        
        p = self.predict(test_data)
        loss = p - test_target
        return (np.square(loss).sum())/test_data.shape[0]

def main():

    def split_data(data_set, ratio=0.9):
        N = data_set.data.shape[0]
        i, j = int(N*ratio), int(N*(1 - ratio))
        return data_set.data[:i], data_set.target[:i], data_set.data[:j], data_set.target[:j]

    from sklearn.datasets import load_boston

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
    





