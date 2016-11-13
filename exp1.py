import os, sys, time, json
import numpy as np
from PIL import Image
import nn

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
    model = nn.train(layerSizes, X, y, lmbda)
    print "Model trained in %s seconds" % (time.time() - start_time)
    return model

def testModel(model, X, y):
    nCorrect = 0
    nGuess = np.zeros(nGesture)
    for i in range(0,X.shape[0]):
        guess = nn.infer(X[i], model)
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
    n = 150*150               #100*100
    nHiddenLayer = 3
    hiddenLayerSize = n/20    #
    lmbda = 0.01

    hiddenLayerSizes = []
    for i in range(0,nHiddenLayer):
        hiddenLayerSizes.append(hiddenLayerSize)
    layerSizes = [n] + hiddenLayerSizes + [nGesture]
    
    (X,y) = readImages(rootDir, nSubject, nGesture, n)
    model = trainModel(layerSizes, X, y, lmbda)
    testModel(model, X, y)
