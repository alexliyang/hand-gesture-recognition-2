import numpy as np
import nn

def debugInitialWeights(nIn, nOut):
    W = np.zeros((nOut, nIn + 1))
    return np.reshape(np.sin(range(1,W.size+1)), W.shape) / 10.0

def numericalGradient(theta, layerSizes, X, y, lmbda):
    numgrad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)
    e = 1e-8
    for p in range(0,theta.size):
        perturb[p] = e
        (loss1,_) = nn.nnCostFunctionFlattened(theta - perturb, layerSizes, X, y, lmbda)
        (loss2,_) = nn.nnCostFunctionFlattened(theta + perturb, layerSizes, X, y, lmbda)
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    return numgrad

def debug():
    input_layer_size = 2
    hidden_layer_size = 2
    num_labels = 2
    m = 1

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

    (cost, grad) = nn.nnCostFunctionFlattened(nn_params, [input_layer_size, hidden_layer_size, num_labels], X, y_encoded, 0)
    numgrad = nn.numericalGradient(nn_params, [input_layer_size, hidden_layer_size, num_labels], X, y_encoded, 0)
    print "\n"
    print cost
    print grad
    print numgrad


if __name__ == "__main__":
    debug()
