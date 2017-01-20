import numpy as np

from scipy.stats import mstats

class LinearRegression(object):
    def __init__(self):
        pass
    
    def fit(self, train_data, train_target):
        X = train_data
        Y = train_target
        self.W = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(Y)
        
    def predict(self, test_data):
        return test_data.dot(self.W)
    
    def mse(self, test_data, test_target, prediction = None):
        p = (prediction if prediction else self.predict(test_data))
        loss = p - test_target
        return (np.square(loss).sum())/test_data.shape[0]
    



