import sys
import csv
import math
import numpy as np

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
            var = temp_np_array.var()
            self.vars.append({'mean': mean, 'var': var})
    
    def normalize_data(self):
        self.normalized_data = np.array([[(float(x) - self.vars[i]['mean'])/self.vars[i]['var'] for i, x in enumerate(row[1:-1])] for row in self.raw_data])
    
    def test(self, row, metric, k):
        res = []
        data = [float(x) for x in row[1:-1]]
        
        for class_name, normalized_data in zip([int(x[-1]) for x in self.raw_data], self.normalized_data):
            distance = 0.0
            for i, x in enumerate(data):
                normalized_x = (x - self.vars[i]['mean'])/self.vars[i]['var']
                distance += metric(normalized_x, normalized_data[i])
            res.append([class_name, distance])
        res.sort(lambda a, b: 1 if a[1] > b[1] else -1)
        if row in self.raw_data:
            return self.get_max_freq(res[1:k+1])
        else:
            return self.get_max_freq(res[:k])
        

    def _reducer(self, x, y):
        x[y] = 1 + x.get(y, 0)
        return x
        
    def get_max_freq(self, arr):
        return max(reduce(self._reducer, [x[0] for x in arr], {}))
        
    def L1(self, x, y):
        return math.sqrt(pow(x-y, 2))
    
    def L2(self, x, y):
        return abs(x-y)

def main(train_file_path, test_file_path, k):
    #train_file_path = sys.argv[1]
    #test_file_path = sys.argv[2]

    train_file_csv_reader = csv.reader(open(train_file_path, 'r'))
    test_file_csv_reader = csv.reader(open(test_file_path, 'r'))
    
    knnClassifier = KNNClassifier()
    knnClassifier.train(list(train_file_csv_reader))

    #print nbClassifier.attr_count 
    #print nbClassifier.data_by_class
    #print nbClassifier.gaussian_vars

    test_passed = 0
    test_failed = 0

    for row in test_file_csv_reader:
        nn = knnClassifier.test(row, knnClassifier.L2, k)
        #print nn
        print "actual class: {}, knn: {}".format(row[-1], nn)
        #break
        """if predicted_class == row[-1]:
            test_passed +=1
        else:
            test_failed += 1
        print "actual_class: {}, predicted_class: {}".format(row[-1], predicted_class)
    print "passed: {}, failed: {}, accuracy: {}%".format(test_passed, test_failed, ((test_passed)/float(test_passed + test_failed))*100) 
    return nbClassifier"""
    return knnClassifier

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))