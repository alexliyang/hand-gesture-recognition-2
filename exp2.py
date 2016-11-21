import time
import numpy as np
from imageReader import ImageReader
from keras.preprocessing.image import ImageDataGenerator
from cnn import CNN

if __name__ == "__main__":
    imageReader = ImageReader()
    cnn = CNN(imageReader.nGesture)
    
    (X,y), (Xv,yv) = imageReader.readImages(make1d=False, validationRatio=0.08)
    print len(X), "training data"
    print len(Xv), "test data"

    imageGenerator = ImageDataGenerator()
    
    cnn.train_gen(imageGenerator.flow(X, y, batch_size=16), len(X), Xv, yv)
    
    trainAccuracy = cnn.test(X,y)
    print "Training Accuracy :", trainAccuracy, "%"
    valAccuracy = cnn.test(Xv,yv)
    print "Test Accuracy :", valAccuracy, "%"

    #imageGenerator = ImageDataGenerator()
    #data = imageGenerator.flow_from_directory(directory='cropp', target_size=(100,100), color_mode='grayscale')
    #test = imageGenerator.flow_from_directory(directory='ASLval', target_size=(100,100), color_mode='grayscale')

    #cnn.train_gen(data, test)
    
    #trainAccuracy = cnn.test_gen(data, 11296)
    #print "Training Accuracy :", trainAccuracy, "%"
    #valAccuracy = cnn.test(test, 1248)
    #print "Test Accuracy :", valAccuracy, "%"
