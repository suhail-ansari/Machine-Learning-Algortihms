import numpy as np
import random
import sys
import matplotlib.pyplot as plt

def convereged(new_mu, old_mu):
    return np.all(np.sort(new_mu) == np.sort(old_mu))
    
def recalculate_mu(clusters):
    mu = np.zeros(shape=(len(clusters.keys()), len(clusters.values()[0][0])))
    for k, points in clusters.items():
        if len(points) > 0:
            mu[k] = np.mean(points, axis=0)
    return mu

def calculate_center(data, mu, k):
    clusters = {i:[] for i in range(k)}
    for point in data:
        min_distance = sys.maxint
        best_k = 0
        for i, _muk in enumerate(mu):
            distance = np.linalg.norm(point - _muk)**2
            if distance < min_distance:
                min_distance = distance
                best_k = i
        clusters[best_k].append(point)
    new_mu = recalculate_mu(clusters)
    return new_mu, clusters

def k_means(data, k, label="", plot="False"):
    old_mu = np.array(random.sample(data, k))
    new_mu, clusters = calculate_center(data, old_mu, k)
    while not convereged(old_mu, new_mu):
        old_mu = new_mu
        new_mu, clusters = calculate_center(data, old_mu, k)
    if plot:
        colors = ['r', 'g', 'b', 'y', 'c']

        for k, points in clusters.items():
            np_points = np.array(points)
            plt.scatter(np_points[:, 0], np_points[:, 1], color=colors[k])
            plt.scatter(new_mu[k][0], new_mu[k][1], marker="^", color=colors[k], edgecolors='black', lw=1, s=50)
        plt.savefig("./images/{}-{}.png".format(label, k + 1))
        plt.close()

    return new_mu, clusters

    