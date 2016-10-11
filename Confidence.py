from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from scipy.signal import medfilt as med_filter
from scipy.ndimage.filters import gaussian_filter
import cv2, os, sys
import scipy.misc
from multiprocessing import Pool
import os

def normalise_to_grayscale(old_array, min, max):
    new_array = np.zeros(old_array.shape)
    for row_idx, row in enumerate(old_array):
        for elm_idx, elm in enumerate(row):
            new_array[row_idx][elm_idx] = (elm - min) * 200 / float(max - min)
    return new_array

def visualise(image_path, output_path):
    im = Image.open(image_path)

    image_array = np.array(im)

    max_val = np.amax(image_array)
    min_val = np.amin(image_array)

    depth_array = normalise_to_grayscale(image_array, min_val, max_val)
    print "save", output_path
    scipy.misc.imsave(output_path, image_array)


def walk_gesture_folder(rootdir, subject_folder, gesture_folder, output_dir="conf_visual/"):
    gesture_dir = os.path.join(rootdir, subject_folder, gesture_folder)
    os.makedirs(os.path.join(output_dir, subject_folder, gesture_folder))
    print "Reading folder", gesture_dir
    results_pool = []
    pool = Pool(processes=40)
    for image_file in os.listdir(gesture_dir):
        image_path = os.path.join(gesture_dir, image_file)
        if os.path.isfile(image_path):
            if not "confi" in image_path or not image_path.endswith(".png"):
                continue

            image_path = os.path.join(rootdir, subject_folder, gesture_folder, image_file)
            output_path = os.path.join(output_dir, subject_folder, gesture_folder, image_file)
            results_pool.append(pool.apply_async(visualise, (image_path, output_path,), ))
            visualise(image_path, output_path)

    pool.close()
    pool.join()

if __name__ == "__main__":
    rootdir="SSF/"
    outputdir="conf_visual"

    print "processing images for subject:", sys.argv[1], "gesture:", sys.argv[2]

    subject_folder = 'ssf14-' + sys.argv[1] + '-depth/'

    walk_gesture_folder(rootdir, subject_folder, sys.argv[2])

    print "Now:", date
    print "Rename output to:", outputdir + '_' + date
    os.rename(outputdir, outputdir + '_' + date)