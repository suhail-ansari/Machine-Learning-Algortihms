import sys
import csv
import math
import numpy as np

from collections import OrderedDict, Counter

class KNNClassifier(object):
    def __init__(self):
        pass
    
    def train(self, data):
        self.raw_data = data
        
        self.attr_count = len(data[0]) - 2
        self.total_samples = len(data)

        self.processed_data = np.array([[float(x) for x in row[1:-1]] for row in self.raw_data])
        self.calculate_vars()
        self.normalize_data()
    
    def calculate_vars(self):
        self.vars = []
        for i in range(self.attr_count):
            temp_np_array = self.processed_data[:, i]
            mean = temp_np_array.mean()
            var = temp_np_array.std(ddof=0)
            self.vars.append({'mean': mean, 'var': var})#math.sqrt(var)})
    
    def normalize_data(self):
        self.normalized_data = np.array([[(float(x) - self.vars[i]['mean'])/self.vars[i]['var'] for i, x in enumerate(row[1:-1])] for row in self.raw_data])
    
    def test(self, row, metric, k):
        res = []
        data = [float(x) for x in row[1:-1]]
        
        for class_name, normalized_data in zip([int(x[-1]) for x in self.raw_data], self.normalized_data):
            distance = 0.0
            
            normalized_test_data = [((x - self.vars[i]['mean'])/self.vars[i]['var']) for i, x in enumerate(data)]
            distance = metric(normalized_test_data, normalized_data)

            res.append([class_name, distance])

        res.sort(lambda a, b: 1 if a[1] > b[1] else -1)
        if row in self.raw_data:
            return self.get_max_freq(res[1:k+1])
        else:
            return self.get_max_freq(res[:k])
        

    def _reducer(self, x, y):
        if not x.get(y):
            return x
        
    def get_max_freq(self, arr):
        res_distances = {}
        res_counter = Counter()

        for _class, distance in arr:
            if not res_distances.get(_class):
                res_distances[_class] = 0
            res_counter[_class] += 1
            res_distances[_class] += distance
        
        most_common_key, max_count = res_counter.most_common()[0]
        keys_to_check =[]
        for k, v in res_counter.most_common():
            if v < max_count:
                break
            else:
                keys_to_check.append(k)
        
        min_distance = sys.maxint
        pred = None
        for k in keys_to_check:
            if res_distances[k]/float(k) < min_distance:
                min_distance = res_distances[k]/float(k)
                pred = k                
        return k
        
    def L2(self, x, y):
        return math.sqrt(sum(map(lambda (i, j) : pow(i - j, 2), zip(x, y))))
    
    def L1(self, x, y):
        return sum(map(lambda (i, j): abs(i-j), zip(x, y)))

def main(train_file_path, test_file_path, k):
    
    train_file_csv_reader = csv.reader(open(train_file_path, 'r'))
    test_file_csv_reader = csv.reader(open(test_file_path, 'r'))
    
    knnClassifier = KNNClassifier()
    knnClassifier.train(list(train_file_csv_reader))

    test_passed = 0
    test_failed = 0

    for row in test_file_csv_reader:
        nn = knnClassifier.test(row, knnClassifier.L2, k)
        print "actual class: {}, knn: {}".format(row[-1], nn)
        if int(row[-1]) == int(nn):
            test_passed += 1
        else:
            test_failed += 1
    print "accuracy: {}%".format(test_passed*100/float(test_failed + test_passed))

    return knnClassifier

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))