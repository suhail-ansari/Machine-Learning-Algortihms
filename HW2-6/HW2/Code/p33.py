import numpy as np
from sklearn.datasets import load_boston
from utils import split_data

import p33a

def main():
    data_set = load_boston()
    train_data, train_target, test_data, test_target = split_data(data_set)
    
    print "\nPearson's Coeff."
    print "{:^15}|{:^15}|{:^15}".format("Feature Index", "Feature Name", "Pearson's Coeff.")
    print "-"*45

    c = []

    for i in range(train_data.shape[1]):
        attr_vals = train_data[:, i]
        r = np.corrcoef(attr_vals, train_target)
        print "{:^15}|{:^15}|{:^15.7}".format(i + 1, data_set.feature_names[i],abs(r[0][1]))
        c.append([i, abs(r[0][1])])
    selected_features = [i for i, _ in sorted(c, key=lambda z: -z[1])[:4]]
    
    p33a.main(selected_features)

if __name__ == "__main__":
    main()