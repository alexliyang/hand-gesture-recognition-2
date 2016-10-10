import numpy as np
from scipy.optimize import fmin_cg

def sigmoid(z):
    return 1.0 / (1 + np.exp(z))

def sigmoidGradient(z):
    gz = sigmoid(z)
    return gz * (1 - gz)

def nnCostFunction(theta, K, X, y, lmbda):
    m = X.shape[0]
    lmbda = lmbda * 1.0
    
    z = [X]
    a = []
    for k in range(0,K):
        a.append(np.append(np.ones((m, 1)), z[k], axis=1))
        z.append(a[k].dot(theta[k].transpose()))
    a.append(sigmoid(z[-1]))
    
    # compute nn cost
    cost = -1.0/m * (y * np.log(a[-1]) + (1 - y) * np.log(1 - a[-1])).sum()
    regCost = 0
    for k in range(0,K):
        regCost += np.square(theta[k][:,2:]).sum()
    regCost = lmbda/(2.0*m) * regCost    
    nnCost = cost + regCost

    # compute gradient
    d = [a[-1] - y]
    for k in range(1,K):
        d.append(d[-1].dot(theta[-k])[:,1:] * sigmoidGradient(z[-(k+1)]))
    d.reverse()

    thetaGrad = np.array([])
    for k in range(0,K):
        regGrad = np.append(np.zeros((theta[k].shape[0],1)), lmbda/m * theta[k][:,1:], axis=1)
        thetaGrad = np.append(thetaGrad, 1.0/m * d[k].transpose().dot(a[k]) + regGrad)

    return (nnCost, thetaGrad)

def nnCostFunctionFlattened(theta_flat, layerSizes, X, y, lmbda):
    K = len(layerSizes) - 1
    theta = []
    thetaStart = 0
    for k in range(0,K):
        thetaEnd = thetaStart + layerSizes[k+1] * (layerSizes[k]+1)
        theta.append(np.reshape(theta_flat[thetaStart:thetaEnd], (layerSizes[k+1], (layerSizes[k]+1))))
        thetaStart = thetaEnd
    
    (cost, thetaGrad) = nnCostFunction(theta, K, X, y, lmbda)
    return (cost, thetaGrad)

def backPropagation(layerSizes, K, X, y, lmbda):
    # init randomize thetas
    # call fmin_cg passing nnCostFunctionFlattened
    # return trained model
    return 0

def infer(input, theta):
    # forward propagation
    return 0

def debugInitialWeights(nIn, nOut):
    W = np.zeros((nOut, nIn + 1))
    return np.reshape(np.sin(range(1,W.size+1)), W.shape, 'F') / 10.0

def numericalGradient(theta, X, y, lmbda):
    numgrad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)
    e = 1e-4
    for p in range(0,theta.size):
        perturb[p] = e
        (loss1,_) = nnCostFunctionFlattened(theta - perturb, [3,5,3], X, y, lmbda)
        (loss2,_) = nnCostFunctionFlattened(theta + perturb, [3,5,3], X, y, lmbda)
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    return numgrad

def debug():
    input_layer_size = 3;
    hidden_layer_size = 5;
    num_labels = 3;
    m = 5;

    Theta1 = debugInitialWeights(input_layer_size, hidden_layer_size)
    Theta2 = debugInitialWeights(hidden_layer_size, num_labels)
    X  = debugInitialWeights(input_layer_size - 1, m)
    y  = np.array([1,2,0,1,2])

    print Theta1
    print Theta2
    print X
    print y

    y_encoded = []
    for i in range(0,m):
        t = np.zeros(num_labels)
        t[y[i]] = 1
        y_encoded.append(t)

    nn_params = np.append(Theta1, Theta2)

    (cost, grad) = nnCostFunctionFlattened(nn_params, [3,5,3], X, np.array(y_encoded), 0)
    numgrad = numericalGradient(nn_params, X, np.array(y_encoded), 0)

    print cost
    print cost.shape
    print "\n"
    print grad
    print grad.shape
    print "\n"
    print numgrad
    print numgrad.shape

if __name__ == "__main__":
    debug()
