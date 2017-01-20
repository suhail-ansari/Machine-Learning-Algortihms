import numpy as np
import random
import kmeans 

from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys

class EMAlgorithm(object):
    def __init__(self, data, k, thresh=10e-04):
        self.data = data
        self.k = k
        self.N = self.data.shape[0]
        self.D = self.data.shape[1]

        self.thresh = thresh

        self.old_ll = 0.0
        self.new_ll = 0.0
        
        self.mus = np.array(random.sample(data, k))
        self.pies = np.array([1.0]*self.k)/self.k
        self.covs = np.array([np.cov(data[:, 0], data[:, 1])]*k)
    
    def estep(self):
        w = np.zeros((self.N, self.k))
        for _k in range(self.k):
            for i in range(self.N):
                w[i, _k] = self.pies[_k] * mvn.pdf(self.data[i], mean=self.mus[_k], cov=self.covs[_k])
        
        for i in range(self.N):
            w[i] = w[i]/w[i].sum()
        self.w = w
    
    def mstep(self):
        for _k in range(self.k):
            pi = 0.0
            mu = np.zeros(self.D)
            for i in range(self.N):
                pi = pi + self.w[i, _k]
                mu = mu +  (self.w[i, _k] * self.data[i])
            self.pies[_k] = pi/self.N
            self.mus[_k] = mu/np.sum(self.w[:, _k])

        for _k in range(self.k):
            cov = np.zeros((self.D, self.D))
            for i in range(self.N):
                _t = (self.data[i]- self.mus[_k]).reshape(self.D, 1)
                cov += self.w[i, _k] * np.dot(_t, _t.T)
            self.covs[_k] = (cov/self.w[:, _k].sum())

    def fit(self, num_iters=100):
        res = []
        for _iter in range(num_iters):
            self.estep()
            self.mstep()
            self.calculate_new_ll()
            
            if self.converged():
                break
            
            res.append([self.new_ll, _iter])
            self.old_ll = self.new_ll

        return np.array(res), self.mus, self.covs, self.pies

    def calculate_new_ll(self):
        new_ll = 0.0
        for i in range(self.N):
            _sum = 0.0
            for _k in range(self.k):
                _sum += (self.pies[_k] * mvn.pdf(self.data[i], mean=self.mus[_k], cov=self.covs[_k]))
            new_ll += np.log(_sum)
        self.new_ll = new_ll
    
    def converged(self):
        return (np.abs(self.new_ll - self.old_ll) < self.thresh)

# ------ Utiliy Function ---------

def predict(data, k, res):
    colors = ['r', 'g', 'b', 'y', 'c']
    for _k in range(k):
        mask = res == _k
        np_points = data[mask]
        plt.scatter(np_points[:, 0], np_points[:, 1], color=colors[_k])
    plt.savefig('./images/EM-Scatterplot.png')
    plt.close()

def print_mean(means):
    print "\nMeans from best run:"
    for _k, mean in enumerate(means):
        print "K={}  ".format(_k + 1), mean

def print_covariance(covs):
    print "\nCovariance matrix from best run:"
    for _k, cov in enumerate(covs):
        print "K={}".format(_k + 1), cov
# ------ Utiliy Function ---------



def main():
    data = np.genfromtxt('./hw5_blob.csv', delimiter=',')
    colors = ['r', 'g', 'b', 'y', 'c']
    legends = []
    
    max_ll = -sys.maxint
    best_mean = None
    best_cov = None
    best_pies = None
    
    for i in range(5):
        em = EMAlgorithm(data, 3)
        res, mus, covs, pies = em.fit(100)
        
        best_ll_in_this_case = max(res, key=lambda x: x[0])[0]
        if best_ll_in_this_case > max_ll:
            max_ll = best_ll_in_this_case
            best_mean = mus
            best_cov = covs
            best_pies = pies
        
        plt.plot(res[:, 1], res[:, 0], color=colors[i])
        patch = mpatches.Patch(color=colors[i], label='{}th run'.format(i + 1))
        legends.append(patch)
    
    plt.legend(handles=legends, loc=4)
    plt.savefig('./images/LogLikelihood.png')
    plt.close()

    indices = []
    last_index = 0
    for pi in best_pies:
        end_index = last_index + int(data.shape[0] * pi)
        end_index = end_index if end_index <= data.shape[0] else data.shape[0]
        indices.append([last_index, end_index])
        last_index = end_index + 1
    
    assignment = np.zeros(data.shape[0])
    for _k, partition in enumerate(indices):
        assignment[partition[0]:partition[1] + 1] = _k
    
    predict(data, 3, assignment)
    print_mean(best_mean)
    print_covariance(best_cov)

if __name__ == "__main__":
    main()