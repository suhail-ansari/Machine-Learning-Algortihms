import numpy as np

def gen_samples(N=100, n=10, mu=0, sigma_sq=0.1):
    all_samples = []
    for i in range(N):
        x = np.random.uniform(-1, 1, n)
        epsilon = np.random.normal(mu, pow(sigma_sq, 0.5), n)
        y = (2*x) + epsilon
        s = np.append(x.reshape(n, 1), y.reshape(n, 1), axis=1)
        all_samples.append(s) 
    return np.array(all_samples)

def expand_features(original_feature, degree):
    expanded_feature_set = original_feature.copy()
    
    if len(expanded_feature_set.shape) != 2:
        expanded_feature_set = expanded_feature_set.reshape(original_feature.shape[0], 1)
    
    for i in range(2, degree):
        new_column = np.power(original_feature, i).reshape(original_feature.shape[0], 1)
        expanded_feature_set = np.append(expanded_feature_set, new_column, axis=1)
    
    return expanded_feature_set