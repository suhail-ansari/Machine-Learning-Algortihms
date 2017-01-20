import numpy as np
from sklearn.datasets import load_boston
import LinearRegression as LR

def main():

    data_set = load_boston()

    train_data, train_target, test_data, test_target = LR.split_data(data_set)
    num_features = train_data.shape[1]

    new_train_data = train_data.copy()
    new_test_data = test_data.copy()
    
    for i in range(num_features):
        for j in range(i, num_features):
            new_column = train_data[:, i] * train_data[:, j]
            new_column = new_column.reshape(new_column.shape[0], 1)
            new_train_data = np.append(new_train_data, new_column, 1)

            new_test_column = test_data[:, i] * test_data[:, j]
            new_test_column = new_test_column.reshape(new_test_column.shape[0], 1)
            new_test_data = np.append(new_test_data, new_test_column, 1)
    
    lr = LR.LinearRegression()
    lr.fit(new_train_data, train_target)
    
    mse_test = lr.mse(new_test_data, test_target)
    mse_train = lr.mse(new_train_data, train_target)

    print "\nSol. 3.4"
    print "Linear Regression"
    print "{:^15}|{:^15}".format("Input Data", "MSE")
    print "-"*30
    print "{:^15}|{:^15.7}".format("test_data", mse_test)
    print "{:^15}|{:^15.7}".format("train_data", mse_train)
    print "\n"

if __name__ == "__main__":
    main()