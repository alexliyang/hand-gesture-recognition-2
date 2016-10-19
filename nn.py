import os, sys, time
import numpy as np
from scipy.optimize import fmin_cg
from PIL import Image
import cv2

def sigmoid(z):
    return 1.0 / (1 + np.exp(z))

def sigmoidGradient(z):
    gz = sigmoid(z)
    return gz * (1 - gz)

def forwardProp(theta, X):
    m = X.shape[0]
    
    z = [X]
    a = []
    for k in range(0,len(theta)):
        zk = z[k]
        if k>0: zk = sigmoid(zk)
        a.append(np.append(np.ones((m, 1)), zk, axis=1))
        z.append(a[k].dot(theta[k].transpose()))
    a.append(sigmoid(z[-1]))
    
    return (z,a)

def J(theta, X, y, lmbda):
    lmbda = lmbda * 1.0
    m = X.shape[0]
    (z,a) = forwardProp(theta, X)

    cost = -1.0/m * (y * np.log(a[-1]) + (1 - y) * np.log(1 - a[-1])).sum()
    
    regCost = 0
    for k in range(0,len(theta)):
        regCost += np.square(theta[k][:,1:]).sum()
    regCost = lmbda/(2.0*m) * regCost    
    
    nnCost = cost + regCost
    print nnCost
    return nnCost

def JPrime(theta, X, y, lmbda):
    lmbda = lmbda * 1.0
    m = X.shape[0]
    (z,a) = forwardProp(theta, X)
    
    # back propagation
    d = [a[-1] - y]
    for k in range(1,len(theta)):
        d.append(d[-1].dot(theta[-k])[:,1:] * sigmoidGradient(z[-(k+1)]))
    d.reverse()

    thetaGrad = np.array([])
    for k in range(0,len(theta)):
        regGrad = np.append(np.zeros((theta[k].shape[0],1)), lmbda/m * theta[k][:,1:], axis=1)
        thetaGrad = np.append(thetaGrad, 1.0/m * d[k].transpose().dot(a[k]) + regGrad) * -1
    print thetaGrad
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
    return J(theta, X, y, lmbda)

def JPrimeFlattened(theta_flat, layerSizes, X, y, lmbda):
    theta = reshapeTheta(theta_flat, layerSizes)
    return JPrime(theta, X, y, lmbda)

def train(layerSizes, X, y, lmbda):
    args = (layerSizes, X, y, lmbda)
    def f(x, *args):
        layerSizes, X, y, lmbda = args
        return JFlattened(x, layerSizes, X, y, lmbda)
    def fPrime(x, *args):
        layerSizes, X, y, lmbda = args
        return JPrimeFlattened(x, layerSizes, X, y, lmbda)

    initEpsilon = 0.12
    theta = np.array([])
    K = len(layerSizes) - 1
    for k in range(0,K):
        theta = np.append(theta, np.random.rand(layerSizes[k+1], layerSizes[k]+1) * 2.0 * initEpsilon - initEpsilon)

    #thetaOpt = fmin_cg(f, theta, maxiter=40, args=args)
    thetaOpt = fmin_cg(f, theta, fprime=fPrime, maxiter=40, args=args)
    return reshapeTheta(thetaOpt, layerSizes)

def iforwardProp(theta, X):
    z = [X]
    a = []
    for k in range(0,len(theta)):
        zk = z[k]
        if k>0: zk = sigmoid(zk)
        a.append(np.append(np.array([1]), zk))
        z.append(a[k].dot(theta[k].transpose()))
    a.append(sigmoid(z[-1]))
    
    return (z,a)

def infer(input, theta):
    (_,a) = iforwardProp(theta, input)
    return np.argmax(a[-1])

def readImages(rootDir, nSubject, nGesture, n):
    xs = []
    ys = []

    iSubject = 0
    for subjectFolder in os.listdir(rootDir):
        subjectDir = os.path.join(rootDir, subjectFolder)
        if not os.path.isdir(subjectDir):
            continue

        iGesture = 0
        for gestureFolder in os.listdir(subjectDir):
            gestureDir = os.path.join(subjectDir, gestureFolder)
            if not os.path.isdir(gestureDir):
                continue

            y = np.zeros(nGesture)
            y[iGesture] = 1

            for imageFile in os.listdir(gestureDir):
                imagePath = os.path.join(gestureDir, imageFile)
                if not os.path.isfile(imagePath) or not imagePath.endswith(".png"):
                    continue

                image = np.array(Image.open(imagePath)).flatten()
                if not image.size == n:
                    print "Skip", imagePath, "of size", image.size 
                    continue 
                xs.append(image)
                ys.append(y)
            
            iGesture = iGesture + 1
            if iGesture >= nGesture:
                break

        iSubject = iSubject + 1
        if iSubject >= nSubject:
            break

    X = np.vstack(xs)
    y = np.vstack(ys)

    return (X,y)

def trainModel(layerSizes, X, y, lmbda):
    start_time = time.time()
    model = train(layerSizes, X, y, lmbda)
    print "Model trained in %s seconds" % (time.time() - start_time)
    return model

def testModel(model, X, y):
    nCorrect = 0
    nGuess = np.zeros(nGesture)
    for i in range(0,X.shape[0]):
        guess = infer(X[i], model)
        nGuess[guess] = nGuess[guess] + 1
        print "Guess", i, ":", guess, "Actual :", np.argmax(y[i])
        nCorrect = nCorrect + y[i][guess]
    print "nGuesses :", nGuess
    print nCorrect, "correct out of", X.shape[0]
    print "Accuracy :", nCorrect * 1.0 / X.shape[0] * 100.0, "%"

if __name__ == "__main__":
    rootDir = 'processed'
    nSubject = 1
    nGesture = 4
    n = 150*150
    nHiddenLayer = 3
    hiddenLayerSize = n/20
    lmbda = 0.01

    hiddenLayerSizes = []
    for i in range(0,nHiddenLayer):
    	hiddenLayerSizes.append(hiddenLayerSize)
    layerSizes = [n] + hiddenLayerSizes + [nGesture]
    
    (X,y) = readImages(rootDir, nSubject, nGesture, n)
    model = trainModel(layerSizes, X, y, lmbda)
    testModel(model, X, y)

    f = open("savedModel", "w")
    json.dump(model, f)
    f.close()