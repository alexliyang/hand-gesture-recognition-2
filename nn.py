import numpy as np
from scipy.optimize import fmin_cg

def sigmoid(z):
    return 1 / (1 + np.exp(z))

def sigmoidGradient(z):
    gz = sigmoid(z)
    return gz * (1 - gz)

def nnCostFunction(theta, K, X, y, lmbda):
    m = X.shape[0]
    
    z = [X]
    a = []
    for k in range(0,K):
        a.append(np.append(np.ones((m, 1)), z[k], axis=1))
        z.append(a[k].dot(theta[k].transpose()))
    a.append(sigmoid(z[-1]))
    
    # compute nn cost
    cost = -1/m * (y * np.log(a[-1]) + (1 - y) * np.log(1 - a[-1])).sum()
    
    regCost = 0
    for k in range(0,K):
        regCost += (theta[k][:,2:] ^ 2).sum()
    regCost = lmbda/(2*m) * 
    
    # compute gradient
    d = [a[-1] - y]
    for k in range(1,K):
        d.append(d[-1].dot(theta[-k])[:,2:] * sigmoidGradient(z[-(k+1)]))

    d.reverse()
    thetaGrad = []
    for k in range(0,K):
        regGrad = np.append(np.zeros(), lmda/m * theta[k][:,2:])
        thetaGrad.append(1/m * d[k].transpose().dot(a[k]) + regGrad)
    
    return (cost + reg_cost, thetaGrad);

def nnCostFunctionFlattened(theta_flat, layerSizes, K, X, y, lmbda):
    theta = []
    thetaStart = 0
    for k in range(0,K):
        thetaEnd = layerSizes[k+1] * (layerSizes[k]+1)
        theta.append(np.reshape(theta_flat[thetaStart:thetaEnd], (layerSizes[k+1], (layerSizes[k]+1))))
        thetaStart = thetaEnd + 1

    (cost, theta_grad) = nnCostFunction(theta, K, X, y, lmbda)
    return theta_grad.flatten()

def backPropagation(layerSizes, K, X, y, lmbda):
    # init randomize thetas
    # call fmin_cg passing nnCostFunctionFlattened
    # return trained model

def infer(input, theta):
    # forward propagation

if __name__ == "__main__":
    # test methods