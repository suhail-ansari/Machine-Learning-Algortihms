import hw_utils as ml_utils

from datetime import datetime 

def main():

    start = datetime.now()

    print "Loading Data..."
    X_tr, y_tr, X_te, y_te = ml_utils.loaddata('./MiniBooNE_PID.txt')
    print X_tr.shape, y_tr.shape
    """
    print "Normalizing Data..."
    nX_tr, nX_te = ml_utils.normalize(X_tr, X_te)

    print "Starting Training..."
    linear_activations(nX_tr, y_tr, nX_te, y_te)
    sigmoid_activations(nX_tr, y_tr, nX_te, y_te)
    relu_activations(nX_tr, y_tr, nX_te, y_te)
    l2_regularization(nX_tr, y_tr, nX_te, y_te)

    best_reg_coeff = early_stopping_l2_regularization(nX_tr, y_tr, nX_te, y_te)
    print "\nbest_reg_coeff: {}\n".format(best_reg_coeff)

    best_decay = SGD_with_weight_decay(nX_tr, y_tr, nX_te, y_te, din=50, dout=2)
    print "\nbest_decay: {}\n".format(best_decay)

    best_momentum = momentum_fn(nX_tr, y_tr, nX_te, y_te, best_decay, din=50, dout=2)
    print "\nbest_momentum: {}\n".format(best_momentum)

    combination(nX_tr, y_tr, nX_te, y_te, best_reg_coeff, best_decay, best_momentum, din=50, dout=2)

    grid_search_with_cross_validation(nX_tr, y_tr, nX_te, y_te, din=50, dout=2)

    stop = datetime.now()
    print "Total Script Time: {}s".format((stop - start).total_seconds())
    """
def linear_activations(nX_tr, y_tr, nX_te, y_te, din=50, dout=2):
    
    print "Linear Activations"

    archs_1 = [
         [din, dout], 
         [din, 50, dout], 
         [din, 50, 50, dout], 
         [din, 50, 50, 50, dout]
    ]

    ml_utils.testmodels(nX_tr, y_tr, nX_te, y_te, archs_1, actfn='linear', sgd_lr=1e-3, verbose=0)

    archs_2 = [
        [din, 50, dout],
        [din, 500, dout],
        [din, 500, 300, dout], 
        [din, 800, 500, 300, dout],
        [din, 800, 800, 500, 300, dout]
    ]

    ml_utils.testmodels(nX_tr, y_tr, nX_te, y_te, archs_2, sgd_lr=1e-3, verbose=0)

    print "Linear Activations - END"

def sigmoid_activations(nX_tr, y_tr, nX_te, y_te, din=50, dout=2):

    print "Sigmoid Activations"
    
    archs = [
        [din, 50, dout],
        [din, 500, dout], 
        [din, 500, 300, dout], 
        [din, 800, 500, 300, dout], 
        [din, 800, 800, 500, 300, dout]
    ]

    ml_utils.testmodels(nX_tr, y_tr, nX_te, y_te, archs, actfn='sigmoid', sgd_lr=1e-3, verbose=0)

    print "Sigmoid Activations - END"

def relu_activations(nX_tr, y_tr, nX_te, y_te, din=50, dout=2):

    print "ReLu Activations"
    
    archs = [
        [din, 50, dout],
        [din, 500, dout], 
        [din, 500, 300, dout], 
        [din, 800, 500, 300, dout], 
        [din, 800, 800, 500, 300, dout]
    ]

    ml_utils.testmodels(nX_tr, y_tr, nX_te, y_te, archs, actfn='relu', sgd_lr=5e-4 , verbose=0)

    print "ReLu Activations - END"

def l2_regularization(nX_tr, y_tr, nX_te, y_te, din=50, dout=2):

    print "L2 Regularization"

    archs = [ [din, 800, 500, 300, dout] ]
    reg_coeffs = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]

    ml_utils.testmodels(nX_tr, y_tr, nX_te, y_te, archs, actfn='relu', reg_coeffs=reg_coeffs, sgd_lr=5e-4 , verbose=0)

    print "L2 Regularization - END"
    
def early_stopping_l2_regularization(nX_tr, y_tr, nX_te, y_te, din=50, dout=2):

    print "Early Stopping and L2-regularization"

    archs = [ [din, 800, 500, 300, dout] ]
    reg_coeffs = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]

    architecture, _lambda, decay, momentum, actfn, best_acc = ml_utils.testmodels(nX_tr, y_tr, 
        nX_te, y_te, archs, actfn='relu', reg_coeffs=reg_coeffs, sgd_lr=5e-4, EStop=True, verbose=0)

    print "Early Stopping and L2-regularization - END"

    return _lambda

def SGD_with_weight_decay(nX_tr, y_tr, nX_te, y_te, din=50, dout=2):

    print "SGD with weight decay"
    
    archs = [ [din, 800, 500, 300, dout] ]
    decays = [5e-5, 1e-4, 3e-4, 7e-4, 1e-3]
    architecture, _lambda, decay, momentum, actfn, best_acc = ml_utils.testmodels(nX_tr, y_tr, nX_te, y_te, archs, 
        actfn='relu', last_act='softmax', reg_coeffs=[5e-7],
        num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decays=decays, sgd_moms=[0.0], 
        sgd_Nesterov=False, EStop=False, verbose=0)
        
    print "SGD with weight decay - END"
    return decay

def momentum_fn(nX_tr, y_tr, nX_te, y_te, best_decay, din=50, dout=2):
    
    print "momentum"
    
    archs = [ [din, 800, 500, 300, dout] ]
    decays = [1e-5, 5e-5, 1e-4, 3e-4, 7e-4, 1e-3]
    architecture, _lambda, decay, momentum, actfn, best_acc = ml_utils.testmodels(nX_tr, y_tr, 
        nX_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs=[0.0],
        num_epoch=50, batch_size=1000, sgd_lr=1e-5, sgd_decays=[best_decay], sgd_moms= [0.99, 0.98, 0.95, 0.9, 0.85], 
        sgd_Nesterov=True, EStop=False, verbose=0)
        
    print "momentum - END"

    return momentum

def combination(nX_tr, y_tr, nX_te, y_te, best_reg_coeff, best_decay, best_momentum, din=50, dout=2):
    
    print "best combination"
    
    archs = [ [din, 800, 500, 300, dout] ]
     
    ml_utils.testmodels(nX_tr, y_tr, nX_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs=[best_reg_coeff],
        num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decays=[best_decay], sgd_moms= [best_momentum], 
        sgd_Nesterov=True, EStop=True, verbose=0)
        
    print "best combination - END"

def grid_search_with_cross_validation(nX_tr, y_tr, nX_te, y_te, din=50, dout=2):
    
    print "Grid search with cross-validation"

    archs = [
        [din, 50, dout], 
        [din, 500, dout], 
        [din, 500, 300, dout], 
        [din, 800, 500, 300, dout], 
        [din, 800, 800, 500, 300, dout]
    ]

    reg_coeffs = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
    decays = [1e-5, 5e-5, 1e-4]
    
    ml_utils.testmodels(nX_tr, y_tr, nX_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs=reg_coeffs,
        num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decays=decays, sgd_moms= [0.99], 
        sgd_Nesterov=True, EStop=True, verbose=0)
    
    print "Grid search with cross-validation - END"
    

if __name__ == "__main__":
    main()