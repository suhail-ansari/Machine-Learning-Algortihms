import numpy as np
import math

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def main():
    data = np.genfromtxt('./hw5_circle.csv', delimiter=',')
    k = 2
    n, d = data.shape
    
    assignments = np.random.randint(0, k, n) 
    distances = np.zeros((n, k))

    new_data = np.empty((n, 3))
    new_data[:, 0] = data[:, 0]
    new_data[:, 1] = data[:, 1]
    new_data[:, 2] = 2 * (np.square(data[:, 1])  + np.square(data[:, 0]))

    max_iters = 1000

    for _iter_num in range(max_iters):
        for _k in range(k):
            mask = (assignments == _k)
            points_in_cluster = new_data[mask]
            center = np.mean(points_in_cluster, axis=0)
            
            for i, point in enumerate(new_data):
                distances[i, _k] = (np.linalg.norm(new_data[i] - center))**2
        
        new_assignments = distances.argmin(axis=1)
                    
        if new_assignments.sum() == assignments.sum():
            break
        assignments = new_assignments 
    
    colors = ['r', 'g']

    for _k in range(k):
        mask = (assignments == _k)
        points = data[mask]
        plt.scatter(points[:, 1], points[:, 0], color=colors[_k])
    plt.savefig('./images/Kernel-Kmeans-2.png')
    plt.close()
    

if __name__ == "__main__":
    main()