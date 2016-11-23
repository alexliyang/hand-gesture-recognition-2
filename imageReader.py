import os, sys
import numpy as np
from PIL import Image
from skimage.feature import hog
import matplotlib.pyplot as plt

def extract_hog_feature(input_img_path):
    # input_img_path = 'testPCA.png'

    image = Image.open(input_img_path)

    depth_array = np.array(image)

    fd = hog(image, orientations=360, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=False)
    # print len(fd)
    threshold = np.mean(fd) * 1.5
    # print threshold
    fd = np.reshape(fd, (360, 36))
    # print fd.shape
    fd = np.where(fd > threshold, fd, 0)

    fd = np.sum(fd, axis=1)
    return fd

class ImageReader:
    def __init__(self):
        self.rootDir = 'final'
        self.n = 100*100
        self.nGesture = 28
    
    def readImages(self, make1d=True, validationRatio=0.1):
        xs = []
        ys = []
        print "root is:", self.rootDir

        iGesture = 0
        for gestureFolder in os.listdir(self.rootDir):
            print "Processing", gestureFolder
            gestureDir = os.path.join(self.rootDir, gestureFolder)
            if not os.path.isdir(gestureDir):
                continue

            y = np.zeros(self.nGesture)
            y[iGesture] = 1

            for imageFile in os.listdir(gestureDir):
                imagePath = os.path.join(gestureDir, imageFile)
                if not os.path.isfile(imagePath) or not imagePath.endswith(".png"):
                    continue

                # image = np.array(Image.open(imagePath))
                if make1d: 
                    image = extract_hog_feature(imagePath)
                    self.n = 360
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
