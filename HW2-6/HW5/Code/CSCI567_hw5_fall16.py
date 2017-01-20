import numpy as np
from kmeans import k_means
import kernel_kmeans
import em_algorithm


import os

def main():

    print "\n>> Since, matplotlib has inconsistent behaviour, I am saving all the generated plots in the directory './images'\n"

    if not os.path.exists('./images'):
        os.mkdir('./images')
    
    print "> Running K-Means for blob_data\n\n"
    blob_data = np.genfromtxt('./hw5_blob.csv', delimiter=',')
    for k in [2, 3, 5]:
        clusters = k_means(blob_data, k, "blob", True)
    
    print "> Running K-Means for circle_data\n\n"
    circle_data = np.genfromtxt('./hw5_circle.csv', delimiter=',')
    for k in [2, 3, 5]:
        clusters = k_means(circle_data, k, "circle", True)
    
    print "> Running Kernel K-Means for circle_data\n\n"
    kernel_kmeans.main()

    print "> Running EM Algorithm for blob_data\n\n"
    em_algorithm.main()

if __name__ == "__main__":
    main()