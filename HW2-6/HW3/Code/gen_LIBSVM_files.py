import numpy as np
import scipy
import csv

TRAIN_FEATURES = "input/phishing-train-features.txt"
TRAIN_LABELS = "input/phishing-train-label.txt"
TEST_FEATURES = "input/phishing-test-features.txt"
TEST_LABELS = "input/phishing-train-label.txt"

COLUMNS_TO_EXPAND = [2, 7, 8, 14, 15, 26, 29]

def txt_np_array(file_name, expand=False, column_to_expand=None):
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        res = []
        for row in reader:
            feature_row = []
            for i, item in enumerate(row):
                if expand:
                    if i in column_to_expand:
                        item = float(item)
                        v = [
                            (1 if int(item) == -1 else 0),
                            (1 if int(item) == 0 else 0),
                            (1 if int(item) == 1 else 0)
                        ]
                        feature_row.extend(v)
                    else:
                        feature_row.append(0 if int(item) == -1 else 1)
                else:
                    feature_row.append(0 if int(item) == -1 else 1)
                    #feature_row.append(int(item))
            res.append(feature_row)
        return np.array(res)

def save_to_file(arr, file_name):
    with open(file_name, 'w') as f:
        for row in arr:
            row_str = ""
            for i, item in enumerate(row):
                if i == 0:
                    row_str = "+1" if item == 1 else "-1"
                else:
                    row_str+= "\t{}:{}".format(i, item)
            row_str += "\n"
            f.write(row_str)

def main():
    columns_to_expand = [(x - 1) for x in COLUMNS_TO_EXPAND]

    ex_train_features = txt_np_array(TRAIN_FEATURES, True, columns_to_expand)
    train_features = txt_np_array(TRAIN_FEATURES)
    train_label = txt_np_array(TRAIN_LABELS)[0]
    
    
    ex_test_features = txt_np_array(TEST_FEATURES, True, columns_to_expand)
    test_features = txt_np_array(TEST_FEATURES)
    test_label = txt_np_array(TEST_LABELS)[0]

    train_data_to_save = np.append(train_label.reshape(train_label.shape[0], 1), ex_train_features, axis=1)
    test_data_to_save = np.append(test_label.reshape(test_label.shape[0], 1), ex_test_features, axis=1)

    save_to_file(train_data_to_save, 'input/mod-phishing-train.txt')
    save_to_file(test_data_to_save, 'input/mod-phishing-test.txt')
    #np.savetxt('input/mod-phishing-train.txt', train_data_to_save, delimiter=' ', newline='\n', fmt='%d')
    #np.savetxt('input/mod-phishing-test.txt', test_data_to_save, delimiter=' ', newline='\n', fmt='%d')
    

if __name__ == "__main__":
    main()