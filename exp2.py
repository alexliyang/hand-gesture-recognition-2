import time
import numpy as np
from imageReader import ImageReader
from cnn import CNN

if __name__ == "__main__":
    imageReader = ImageReader()
    cnn = CNN(imageReader.nGesture)
    
    (X,y) = imageReader.readImages(make1d=False)
    print len(X), "test data"
    cnn.train(X,y)
    cnn.test(X,y)