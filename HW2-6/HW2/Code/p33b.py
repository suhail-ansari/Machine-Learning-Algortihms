import numpy as np
import LinearRegression as LR

from sklearn.datasets import load_boston
from utils import split_data
from scipy.stats import mstats 

def main():

    data_set = load_boston()

    train_data, train_target, test_data, test_target = split_data(data_set)

    C = []
    for i, col in enumerate(train_data.T):
        X = mstats.zscore(col)
        Y = train_target
        r = np.corrcoef(X, Y)[0][1]
        C.append([i, abs(r)])
    
    feature_with_max_r_target = max(C, key=lambda z: z[1])[0]
    
    lr = LR.LinearRegression()
    lr.fit(train_data[:, [feature_with_max_r_target]], train_data)
    
    featues_inluded = [0, feature_with_max_r_target]
    
    print "Sol. 3.3.b\n"
    print "Selecting the feature with max Correlation Coff. with Target. We get: [{}, {}]\n".format(
        data_set.feature_names[feature_with_max_r_target], feature_with_max_r_target + 1)

    for i in range(3):
        print "Selecting Next feature"
        print "{:^15}|{:^15}".format("Feature", "abs(r(residue, attr))")
        print "-"*30
        C_against_residue = []
        for j in range(train_data.shape[1]):
            if j not in featues_inluded:
                lr = LR.LinearRegression()
                lr.fit(train_data[:, featues_inluded], train_target)
                residue = train_target - lr.predict(train_data[:, featues_inluded])

                X = mstats.zscore(train_data[:, j])
                r = np.corrcoef(residue, X)[0, 1]
                C_against_residue.append([j, abs(r)])
        
        for c in C_against_residue:
            print "{:^15}|{:^15.7}".format(data_set.feature_names[c[0]], c[1])
        
        feature_with_max_r_residue = max(C_against_residue, key=lambda z: z[1])[0]
        
        print "\nSelecting Feature with max correlation coeff.: {}, 1-index: {}\n".format(data_set.feature_names[feature_with_max_r_residue],
            feature_with_max_r_residue + 1)    
        
        featues_inluded.append(feature_with_max_r_residue)
        
    print "Selected Features (in-order of selection): [{}], by 1-index: {}".format(", ".join([data_set.feature_names[i] for i in featues_inluded[1:]]),
        [x + 1 for x in featues_inluded[1:]])

    
    lr = LR.LinearRegression()
    lr.fit(train_data[:, featues_inluded], train_target)
    mse_train = lr.mse(train_data[:, featues_inluded], train_target)
    mse_test = lr.mse(test_data[:, featues_inluded], test_target)
    
    print "{:^15}|{:^15}".format("Input Data", "MSE")
    print "-"*30
    print "{:^15}|{:^15.7}".format("test_data", mse_test)
    print "{:^15}|{:^15.7}".format("train_data", mse_train)

if __name__ == "__main__":
    main()