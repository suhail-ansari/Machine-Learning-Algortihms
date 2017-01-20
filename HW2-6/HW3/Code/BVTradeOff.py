import numpy as np
import utils
import math

import LinearRegression as LR

def p(x, var):
    res = 1/(math.sqrt(2 * math.pi * var))
    res *= math.exp(-(x ** 2)/(2 * var))
    return res

def lr_g1(_x, y):
    return np.ones(shape=y.shape)

def lr_g2(_x, y):
    _w = np.ones(shape=y.shape) * y.mean()
    return _w

def linear_regression(N=100, n=10, mu=0, sigma_sq=0.1):
    S = utils.gen_samples(N, n, mu, sigma_sq)
    noise_var = sigma_sq
    
    mses = []
    predictions = {x:[] for x in range(6)}
    
    for i in range(S.shape[0]):
        x = S[i][:, 0]
        y = S[i][:, 1]
        
        sample_mse = [] 
        sample_vars = []
        sample_bias2s = []
        
        for j in range(6):
            
            if j == 0:
                ex = x
                y_hat = lr_g1(x, y)
            elif j == 1:
                ex = x
                y_hat = lr_g2(x, y)
            elif j == 2:
                ex = x.reshape(x.shape[0], 1)
                ex = np.append(np.ones(shape=(ex.shape[0], 1)), ex, axis=1)
                lr = LR.LinearRegression()
                lr.fit(ex, y)
                y_hat = lr.predict(ex)
            else:
                ex = utils.expand_features(x, j)
                ex = np.append(np.ones(shape=(ex.shape[0], 1)), ex, axis=1)
                lr = LR.LinearRegression()
                lr.fit(ex, y)
                y_hat = lr.predict(ex)
            
            predictions[j].append(y_hat.tolist())

            sse = np.sum(np.power(y_hat - y, 2))
            mse = sse/ex.shape[0]
            sample_mse.append(mse)
            
        mses.append(sample_mse)
    mses = np.array(mses)
    
    bias2s = get_bias(predictions, S[:, :, 1].ravel(), n, N)
    vars = get_variance(predictions, S, n, N)

    print "Sample Size (n) = {}".format(n)
    print "{:^20}|{:^20}|{:^20}".format("g(x)", 'Var[y]', "Bias2")
    print "-"*60
    for i in range(6):
        print "{:^20}|{:^20}|{:^20}".format("g{}(x)".format(i + 1), vars[i], bias2s[i])
    print "\n\n"    

    #return mses, vars, bias2s, predictions

def get_variance(predictions, S, n, N):
    vars = [0.0]*len(predictions.keys())
    for i in predictions.keys():
        p = predictions[i]
        means = np.array(p).mean(axis=0)
        p = np.array(p).T
        sum = 0
        for j, row in enumerate(p):
            for k, hx in enumerate(row):
                 sum += np.power(hx - means[j], 2) 
        vars[i] = sum/(N*N)
        vars[1] = 0.0
    return vars

def get_bias(predictions, y, n, N):
    bias = [0.0]*len(predictions.keys())
    E_y_y = y.mean()
    for i in predictions.keys():
        p = predictions[i]
        means = np.asarray(p).mean(axis=0)
        x = 0.0
        for mean in means:
            x += pow(mean - E_y_y, 2)
        bias[i] += x
    bias = [b/N for b in bias]
    return bias

def main():
    print "\n\nLinear Regression\n"
    mu = 0
    sigma_sq = 0.1
    return (linear_regression(100, 10, mu, sigma_sq),
            linear_regression(100, 100, mu, sigma_sq),)
    

if __name__ == "__main__":
    main()