import numpy as np
import sys
import LinearRegression as LR

from sklearn.datasets import load_boston
from utils import split_data
from itertools import permutations

def main():

    data_set = load_boston()

    train_data, train_target, test_data, test_target = LR.split_data(data_set)

    min_MSE = sys.maxint
    min_combo = None

    calculated_combos = []

    for combo in set(permutations(range(train_data.shape[1]), 4)):
        if sorted(combo) not in calculated_combos:
            calculated_combos.append(sorted(combo))
            lr = LR.LinearRegression()
            lr.fit(train_data[:, sorted(combo)], train_target)
            MSE = lr.mse(test_data[:, sorted(combo)], test_target)
            
            if min_MSE > MSE:
                min_MSE = MSE
                min_combo = combo
    
    print "Brute Force"
    print "Best Combination : [{}], by 1-index: {} with MSE = {:.7}".format(
        ", ".join([data_set.feature_names[x] for x in min_combo]), 
        [x + 1 for x in min_combo],
        min_MSE)
    

if __name__ == "__main__":
    main()