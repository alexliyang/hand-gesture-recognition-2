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
        zk = z[k]
        if k>0: zk = sigmoid(zk)
        a.append(np.append(np.ones((m, 1)), zk, axis=1))
        z.append(a[k].dot(theta[k].transpose()))
    a.append(sigmoid(z[-1]))
    
    # compute nn cost
    cost = -1.0/m * (y * np.log(a[-1]) + (1 - y) * np.log(1 - a[-1])).sum()
    regCost = 0
    for k in range(0,K):
        regCost += np.square(theta[k][:,1:]).sum()
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

def reshapeTheta(theta_flat, layerSizes):
    K = len(layerSizes) - 1
    theta = []
    thetaStart = 0
    for k in range(0,K):
        thetaEnd = thetaStart + layerSizes[k+1] * (layerSizes[k]+1)
        theta.append(np.reshape(theta_flat[thetaStart:thetaEnd], (layerSizes[k+1], (layerSizes[k]+1))))
        thetaStart = thetaEnd
    return theta

def nnCostFunctionFlattened(theta_flat, layerSizes, X, y, lmbda):
    K = len(layerSizes) - 1
    theta = reshapeTheta(theta_flat, layerSizes)
    (cost, grad) = nnCostFunction(theta, K, X, y, lmbda)
    return (cost, grad)

def backPropagation(layerSizes, X, y, lmbda):
    args = (layerSizes, X, y, lmbda)
    def f(x, *args):
        layerSizes, X, y, lmbda = args
        return nnCostFunctionFlattened(x, layerSizes, X, y, lmbda)

    initRange = 0.24
    theta = np.array([])
    K = len(layerSizes) - 1
    for k in range(0,K):
        theta = np.append(theta, np.random.rand(layerSizes[k+1], layerSizes[k]+1) * initRange)

    # optimize and return
    thetaOpt = fmin_cg(f, theta, maxiter=400, args=args)
    return reshapeTheta(thetaOpt, layerSizes)

def infer(input, theta):
    # forward propagation
    inference = np.array(input)
    for k in range(0,len(theta)):
        inference = np.append(1, inference).dot(theta[k].transpose())
    return np.argmax(inference)

def debugInitialWeights(nIn, nOut):
    W = np.zeros((nOut, nIn + 1))
    return np.reshape(np.sin(range(1,W.size+1)), W.shape) / 10.0

def numericalGradient(theta, layerSizes, X, y, lmbda):
    numgrad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)
    e = 1e-4
    for p in range(0,theta.size):
        perturb[p] = e
        (loss1,_) = nnCostFunctionFlattened(theta - perturb, layerSizes, X, y, lmbda)
        (loss2,_) = nnCostFunctionFlattened(theta + perturb, layerSizes, X, y, lmbda)
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    return numgrad

def debug():
    input_layer_size = 2;
    hidden_layer_size = 2;
    num_labels = 2;
    m = 1;

    Theta1 = debugInitialWeights(input_layer_size, hidden_layer_size)
    Theta2 = debugInitialWeights(hidden_layer_size, num_labels)
    X  = debugInitialWeights(input_layer_size - 1, m)
    y  = np.array([1])

    print Theta1
    print Theta1.shape
    print Theta2
    print Theta2.shape
    print X
    print X.shape

    y_encoded = []
    for i in range(0,m):
        t = np.zeros(num_labels)
        t[y[i]] = 1
        y_encoded.append(t)
    y_encoded = np.array(y_encoded)
    print y_encoded
    print y_encoded.shape

    nn_params = np.append(Theta1, Theta2)
    print nn_params
    print "\n"

    (cost, grad) = nnCostFunctionFlattened(nn_params, [input_layer_size, hidden_layer_size, num_labels], X, y_encoded, 0)
    numgrad = numericalGradient(nn_params, [input_layer_size, hidden_layer_size, num_labels], X, y_encoded, 0)
    print "\n"
    print cost
    print grad
    print numgrad

if __name__ == "__main__":
    #theta = backPropagation([2,4,4,2], np.array([[0,0],[0,1],[1,0],[1,1],[0,0],[0,1],[1,0],[1,1]]), np.array([[1,0],[0,1],[0,1],[1,0],[1,0],[0,1],[0,1],[1,0]]), 0)
    #print infer([0,0], theta)
    #print infer([0,1], theta)
    #print infer([1,0], theta)
    #print infer([1,1], theta)
    debug()