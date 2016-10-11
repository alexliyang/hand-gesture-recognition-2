import numpy as np
from scipy.optimize import fmin_cg

def sigmoid(z):
    return 1.0 / (1 + np.exp(z))

def sigmoidGradient(z):
    gz = sigmoid(z)
    return gz * (1 - gz)

def forwardProp(theta, K, X):
    m = X.shape[0]
    
    z = [X]
    a = []
    for k in range(0,K):
        zk = z[k]
        if k>0: zk = sigmoid(zk)
        a.append(np.append(np.ones((m, 1)), zk, axis=1))
        z.append(a[k].dot(theta[k].transpose()))
    a.append(sigmoid(z[-1]))
    
    return (z,a)

def J(theta, K, X, y, lmbda):
    lmbda = lmbda * 1.0
    m = X.shape[0]
    (z,a) = forwardProp(theta, K, X)

    cost = -1.0/m * (y * np.log(a[-1]) + (1 - y) * np.log(1 - a[-1])).sum()
    
    regCost = 0
    for k in range(0,K):
        regCost += np.square(theta[k][:,1:]).sum()
    regCost = lmbda/(2.0*m) * regCost    
    
    return cost + regCost

def JPrime(theta, K, X, y, lmbda):
    lmbda = lmbda * 1.0
    m = X.shape[0]
    (z,a) = forwardProp(theta, K, X)
    
    d = [a[-1] - y]
    for k in range(1,K):
        d.append(d[-1].dot(theta[-k])[:,1:] * sigmoidGradient(z[-(k+1)]))
    d.reverse()

    thetaGrad = np.array([])
    for k in range(0,K):
        regGrad = np.append(np.zeros((theta[k].shape[0],1)), lmbda/m * theta[k][:,1:], axis=1)
        thetaGrad = np.append(thetaGrad, 1.0/m * d[k].transpose().dot(a[k]) + regGrad)
    return thetaGrad

def reshapeTheta(theta_flat, layerSizes):
    K = len(layerSizes) - 1
    theta = []
    thetaStart = 0
    for k in range(0,K):
        thetaEnd = thetaStart + layerSizes[k+1] * (layerSizes[k]+1)
        theta.append(np.reshape(theta_flat[thetaStart:thetaEnd], (layerSizes[k+1], (layerSizes[k]+1))))
        thetaStart = thetaEnd
    return theta

def JFlattened(theta_flat, layerSizes, X, y, lmbda):
    theta = reshapeTheta(theta_flat, layerSizes)
    return J(theta, len(layerSizes) - 1, X, y, lmbda)

def JPrimeFlattened(theta_flat, layerSizes, X, y, lmbda):
    theta = reshapeTheta(theta_flat, layerSizes)
    return JPrime(theta, len(layerSizes) - 1, X, y, lmbda)

def train(layerSizes, X, y, lmbda):
    args = (layerSizes, X, y, lmbda)
    def f(x, *args):
        layerSizes, X, y, lmbda = args
        return JFlattened(x, layerSizes, X, y, lmbda)
    def fPrime(x, *args):
        layerSizes, X, y, lmbda = args
        return JPrimeFlattened(x, layerSizes, X, y, lmbda)

    initRange = 0.24
    theta = np.array([])
    K = len(layerSizes) - 1
    for k in range(0,K):
        theta = np.append(theta, np.random.rand(layerSizes[k+1], layerSizes[k]+1) * initRange)

    # optimize and return
    thetaOpt = fmin_cg(f, theta, maxiter=400, args=args)
    return reshapeTheta(thetaOpt, layerSizes)

def infer(input, theta):
    inference = np.array(input)
    for k in range(0,len(theta)):
        inference = np.append(1, inference).dot(theta[k].transpose())
    return np.argmax(inference)

if __name__ == "__main__":
    
