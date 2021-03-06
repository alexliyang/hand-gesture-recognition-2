import time
import numpy as np
from imageReader import ImageReader
import nn

def trainModel(layerSizes, X, y, lmbda, maxIter):
    start_time = time.time()
    model = nn.train(layerSizes, X, y, lmbda, maxIter)
    print "Model trained in %s seconds" % (time.time() - start_time)
    return model

def testModel(model, X, y, nGesture):
    nCorrect = 0
    nGuess = np.zeros(nGesture)
    for i in range(0,X.shape[0]):
        guess = nn.infer(X[i], model)
        nGuess[guess] = nGuess[guess] + 1
        #print "Guess", i, ":", guess, "Actual :", np.argmax(y[i])
        nCorrect = nCorrect + y[i][guess]
    #print nCorrect, "correct out of", X.shape[0]
    print "Accuracy :", nCorrect * 1.0 / X.shape[0] * 100.0, "%"
    print "nGuesses :", nGuess
    
if __name__ == "__main__":
    imageReader = ImageReader()
    nHiddenLayer = 3
    hiddenLayerSize = 1000
    lmbda = 0.01
    maxIter = 400

    hiddenLayerSizes = []
    for i in range(0, nHiddenLayer):
        hiddenLayerSizes.append(hiddenLayerSize)
    layerSizes = [imageReader.n] + hiddenLayerSizes + [imageReader.nGesture]
    
    (X,y), (Xv, yv) = imageReader.readImages()
    print len(X), "training data"
    print len(Xv), "test data"
    model = trainModel(layerSizes, X, y, lmbda, maxIter)
    
    print "nHiddenLayer :", nHiddenLayer
    print "hiddenLayerSize :", hiddenLayerSize
    print "lmbda :", lmbda
    print "maxIter :", maxIter

    print "Training",
    testModel(model, X, y, imageReader.nGesture)

    print "Test",
    testModel(model, Xv, yv, imageReader.nGesture)
