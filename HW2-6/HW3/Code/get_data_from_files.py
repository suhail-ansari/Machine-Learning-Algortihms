import os
import re
from tabulate import tabulate

v_to_l = {6.103515625e-05: '4^{-7}',
 0.000244140625: '4^{-6}',
 0.0009765625: '4^{-5}',
 0.00390625: '4^{-4}',
 0.015625: '4^{-3}',
 0.0625: '4^{-2}',
 0.25: '4^{-1}',
 1: '4^{0}',
 4: '4^{1}',
 16: '4^{2}',
 64: '4^{3}',
 256: '4^{4}',
 1024: '4^{5}',
 4096: '4^{6}',
 16384: '4^{7}'
}

rx = re.compile(r'\[([0-9]*\.?[0-9]*)\]')

def get_data_linear(dir_name):
    file_names = os.listdir(dir_name)
    
    res = []
    for file_name in file_names:
        if "result_" in file_name:
            v = rx.findall(file_name)
            l = v_to_l[float(v[0])]
            t = 0.0
            accuracy = 0.0
            text_lines = open(os.path.join(dir_name, file_name)).read().split("\n")
            for line in text_lines:
                if "real" in line:
                    t = line.split("	")[1].strip()
                elif "Cross Validation Accuracy" in line:
                    accuracy = line.split("=")[1].strip()
            res.append([l, accuracy, t])
    headers = ['C', '3-Fold Cross Validation Accuracy', 'Execution Time']
    print apply("{:^20}|{:^20}|{:^20}".format, headers)
    print "-"*60
    for r in res:
        print apply("{:^20}|{:^20}|{:^20}".format, r)

def get_data_poly(dir_name):
    file_names = os.listdir(dir_name)
    
    res = []
    for file_name in file_names:
        if "result_" in file_name:
            v = rx.findall(file_name)
            #print v
            d = v[1]
            l = v_to_l[float(v[0])]
            t = 0.0
            accuracy = 0.0
            text_lines = open(os.path.join(dir_name, file_name)).read().split("\n")
            for line in text_lines:
                if "real" in line:
                    t = line.split("	")[1].strip()
                elif "Cross Validation Accuracy" in line:
                    accuracy = line.split("=")[1].strip()
            res.append([l, d, accuracy, t])
    
    headers = ['C', 'degree', '3-Fold Cross Validation Accuracy', 'Execution Time']
    print apply("{:^20}|{:^20}|{:^20}|{:^20}".format, headers)
    print "-"*80
    for r in res:
        print apply("{:^20}|{:^20}|{:^20}|{:^20}".format, r)

def get_data_rbf(dir_name):
    file_names = os.listdir(dir_name)
    
    res = []
    for file_name in file_names:
        if "result_" in file_name:
            v = rx.findall(file_name)
            #print v
            g = v_to_l[float(v[1])]
            l = v_to_l[float(v[0])]
            t = 0.0
            accuracy = 0.0
            text_lines = open(os.path.join(dir_name, file_name)).read().split("\n")
            for line in text_lines:
                if "real" in line:
                    t = line.split("	")[1].strip()
                elif "Cross Validation Accuracy" in line:
                    accuracy = line.split("=")[1].strip()
            res.append([l, g, accuracy, t])
    
    headers = ['C', '\gamma', '3-Fold Cross Validation Accuracy', 'Execution Time']
    print apply("{:^20}|{:^20}|{:^20}|{:^20}".format, headers)
    print "-"*80
    for r in res:
        print apply("{:^20}|{:^20}|{:^20}|{:^20}".format, r)

if __name__ == "__main__":
    get_data_linear('output/linear')
    print "\n\n"
    get_data_poly('output/poly')
    print "\n\n"
    get_data_rbf('output/rbf')
