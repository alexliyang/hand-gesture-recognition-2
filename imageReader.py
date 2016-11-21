import os, sys
import numpy as np
from PIL import Image

class ImageReader:
    def __init__(self):
        self.rootDir = 'final'
        self.n = 100*100
        self.nGesture = 28
    
    def readImages(self, make1d=True, validationRatio=0.1):
        xs = []
        ys = []

        iGesture = 0
        for gestureFolder in os.listdir(self.rootDir):
            gestureDir = os.path.join(self.rootDir, gestureFolder)
            if not os.path.isdir(gestureDir):
                continue

            y = np.zeros(self.nGesture)
            y[iGesture] = 1

            for imageFile in os.listdir(gestureDir):
                imagePath = os.path.join(gestureDir, imageFile)
                if not os.path.isfile(imagePath) or not imagePath.endswith(".png"):
                    continue

                image = np.array(Image.open(imagePath))
                if make1d: 
                    image = image.flatten()
                else:
                    image = [image]
                xs.append(image)
                ys.append(y)
            
            iGesture = iGesture + 1
 
        xt = []
        yt = []
        nTrain= len(xs) - int(validationRatio * len(xs))
        permutation = np.random.permutation(len(xs))
        for i in permutation[:nTrain]:
            xt.append(xs[i])
            yt.append(ys[i])
            
        xv = []
        yv = []
        for i in permutation[nTrain:]:
            xv.append(xs[i])
            yv.append(ys[i])

        return (np.asarray(xt), np.asarray(yt)), (np.asarray(xv), np.asarray(yv))
