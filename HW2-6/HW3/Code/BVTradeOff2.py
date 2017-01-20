import numpy as np
import RidgeRegression as RR
import math
import utils

def main():
    N = 100
    n = 100
    mu = 0.0
    sigma_sq = 0.1

    S = utils.gen_samples(N, n, mu, sigma_sq) 
    
    ls = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    predictions = {x:[] for x in ls}

    for i in range(S.shape[0]):
        x = S[i][:, 0]
        y = S[i][:, 1]
        
        ex = utils.expand_features(x, 3)
        ex = np.append(np.ones(shape=(ex.shape[0], 1)), ex, axis=1)
        
        for l in ls:
            rr = RR.RidgeRegression()
            rr.fit(ex, y, l)
            y_hat = rr.predict(ex)
            
            predictions[l].append(y_hat)
    
    #return predictions

    
    Y = S[:, :, 1].ravel()
    E_Y = Y.mean()

    bias = get_bias(predictions, Y, n , N)
    vars = get_variance(predictions, S, n, N)

    """    
    bias = {l:0.0 for l in ls}
    vars = {l:0.0 for l in ls}

    for l in predictions.keys():
        preds = predictions[l]
        for i, sample_prediction in enumerate(preds):
            ED_HD = sum(sample_prediction)/n
            #b = (ED_HD - E_Y)**2
            
            v = 0.0
            b = 0.0
            
            for j, hx in enumerate(sample_prediction):
                v += pow(ED_HD - hx, 2)
                b += pow(ED_HD - 2*S[i, j, 0], 2)
        
            vars[l] += v/n
            bias[l] += b/n
    
    vars = {k:v/N for k, v in vars.items()}
    bias = {k:b/N for k, b in bias.items()}
    """
    
    print "\nRidge Regression"
    print "{:^20}|{:^20}|{:^20}".format("lambda", "Var[y]", "Bias2")
    print "-"*60
    for l in ls:
        print "{:^20}|{:^20}|{:^20}".format(l, vars[l], bias[l])
    

    #import matplotlib.pyplot as plt
    #plt.plot(ls, [vars[l] for l in ls], "--")
    #plt.plot(ls, [bias[l] for l in ls])
    #plt.show()

    #plt.plot(ls, [bias[l] for l in ls])
    #plt.show()
    
    """
    return bias, vars, predictions
    """

def get_variance(predictions, S, n, N):
    vars = {l:0.0 for l in predictions.keys()} 
    
    for i in predictions.keys():
        p = predictions[i]
        means = np.asarray(p).mean(axis=0)
        sum = 0
        for j, sample in enumerate(S):
            for k, row in enumerate(sample):
                sum += np.power(p[j][k] - means[k], 2)
        vars[i] = sum/(N*n)  
    return vars

def get_bias(predictions, y, n, N):
    bias = {l:0.0 for l in predictions.keys()}
    E_y_y = y.mean()
    for i in predictions.keys():
        p = predictions[i]
        means = np.asarray(p).mean(axis=0)
        x = 0.0
        for mean in means:
            x += pow(mean - E_y_y, 2)
        bias[i] += x
    bias = {l: b/N for l, b in bias.items()}
    return bias    

if __name__ == "__main__":
    main()