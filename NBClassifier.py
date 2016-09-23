import sys
import math
import numpy as np
import csv

class NBClassifier(object):
    def __init__(self):
        pass
    def train(self, data):
        """
        @param: list of unprocessed data
        """
        self.raw_data = data
        
        self.attr_count = len(data[0]) - 2
        self.total_samples = len(data)
        
        self.split_data_by_class()
        self.calculate_priors()
        self.calculate_gaussian_vars()
        self.calculates_features_to_ignore()

    def split_data_by_class(self):
        self.data_by_class = {}
        
        for raw_data_row in self.raw_data:
            class_name = raw_data_row[-1]
            data_row = [float(x) for x in raw_data_row[1:-1]]
            
            if class_name not in self.data_by_class:
                self.data_by_class.setdefault(class_name, [])
            self.data_by_class[class_name].append(data_row)
    
    def calculates_features_to_ignore(self):
        self.features_to_ignore = []
        for class_name, gaussian_vars in self.gaussian_vars.items():
            for i, gv in enumerate(gaussian_vars):
                if gv['var'] == 0.0 and i not in self.features_to_ignore:
                    self.features_to_ignore.append(i)


    def calculate_priors(self):
        self.priors = {}
        for class_name, data_rows in self.data_by_class.items():
            prior_prob = float(len(data_rows))/float(self.total_samples)
            self.priors.setdefault(class_name, prior_prob)

    def calculate_gaussian_vars(self):
        self.gaussian_vars = {}
        
        for class_name, data_rows in self.data_by_class.items():
            np_data_rows = np.array(data_rows)
            #print np_data_rows
            for i in range(self.attr_count):
                attr_vals = np_data_rows[:, i]
                mean = attr_vals.mean()
                var = attr_vals.var()
                if class_name not in self.gaussian_vars:
                    self.gaussian_vars.setdefault(class_name, [])
                self.gaussian_vars[class_name].append({'mean': mean, 'var': var})

    def test(self, row, ignore=False, use_numpy_min=False):
        actual_class_name = row[-1]
        row = [float(x) for x in row[1:-1]]

        max_prob = -1.0
        predicted_class = None

        for class_name, gaussian_vars in self.gaussian_vars.items():
            calculated_prob = self.priors[class_name]
            
            for i, feature_val in enumerate(row):
                if ignore:
                    if i not in self.features_to_ignore:
                        if gaussian_vars[i]['var'] > 0.0:

                            mult_1 = 1.0/(math.sqrt(2*math.pi*gaussian_vars[i]['var']))
                            mult_2 = math.exp((-1.0/(2.0*gaussian_vars[i]['var']))*pow((feature_val - gaussian_vars[i]['mean']), 2))
                            calculated_prob *= (mult_1*mult_2)
                        
                        elif use_numpy_min:
                            _var = 2.01032500e-09    
                            mult_1 = 1.0/(math.sqrt(2*math.pi*_var))
                            mult_2 = math.exp((-1.0/(2.0*_var))*pow((feature_val - gaussian_vars[i]['mean']), 2))
                            calculated_prob *= (mult_1*mult_2)
                else:
                    if gaussian_vars[i]['var'] > 0.0:
                        mult_1 = 1.0/(math.sqrt(2*math.pi*gaussian_vars[i]['var']))
                        mult_2 = math.exp((-1.0/(2.0*gaussian_vars[i]['var']))*pow((feature_val - gaussian_vars[i]['mean']), 2))
                        calculated_prob *= (mult_1*mult_2)
                    
                    elif use_numpy_min:
                        _var = 2.01032500e-09    
                        mult_1 = 1.0/(math.sqrt(2*math.pi*_var))
                        mult_2 = math.exp((-1.0/(2.0*_var))*pow((feature_val - gaussian_vars[i]['mean']), 2))
                        calculated_prob *= (mult_1*mult_2)

            if calculated_prob > max_prob or predicted_class == None:
                max_prob = calculated_prob
                predicted_class = class_name
        
        return predicted_class#, max_prob

def main(train_file_path, test_file_path):

    train_file_csv_reader = csv.reader(open(train_file_path, 'r'))
    test_file_csv_reader = csv.reader(open(test_file_path, 'r'))
    
    nbClassifier = NBClassifier()
    nbClassifier.train(list(train_file_csv_reader))

    test_passed = 0
    test_failed = 0

    for row in test_file_csv_reader:
        predicted_class = nbClassifier.test(row)
        if predicted_class == row[-1]:
            test_passed +=1
        else:
            test_failed += 1
        print "actual_class: {}, predicted_class: {}".format(row[-1], predicted_class)
    print "passed: {}, failed: {}, accuracy: {}%".format(test_passed, test_failed, ((test_passed)/float(test_passed + test_failed))*100) 
    return nbClassifier

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])