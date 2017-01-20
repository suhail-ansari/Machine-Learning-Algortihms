import sys

try:
    from svmutil import *
except ImportError:
    sys.path.append('./libsvm/python')
    try:
        from svmutil import *
    except ImportError:
        print "svmutils library not in path. Please make sure to add the libsvm folder in this directory."
        sys.exit()

import gen_LIBSVM_files

def main():
    gen_LIBSVM_files.main()

    C = 4**6
    G = 4**(-4)

    train_y, train_x = svm_read_problem("./input/mod-phishing-train.txt")
    test_y, test_x = svm_read_problem("./input/mod-phishing-test.txt")
    
    p = '-s 0 -c {} -g {} -t 2'.format(C, G)
    model = svm_train(train_y, train_x, p)
    
    p_label, p_acc, p_val = svm_predict(test_y, test_x, model)
    
    print "\n\nSVM, using C = {}, gamma = {}".format(C, G) 
    print "\n", ">> Test Set Accuracy: {}%".format(p_acc[0])

if __name__ == "__main__":
    main()