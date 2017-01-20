import numpy as np

class RidgeRegression(object):
    def fit(self, X, y, alpha=0):
        #X = np.hstack((np.ones((X.shape[0], 1)), X))
        G = alpha * np.eye(X.shape[1])
        G[0, 0] = 1  # Don't regularize bias
        self.params = np.dot(np.linalg.inv(np.dot(X.T, X) + np.dot(G.T, G)),
                             np.dot(X.T, y))

    def predict(self, X):
        return np.dot(X, self.params)