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
import os, math

def normalize_to_gray_scale(old_array, min, max, reverse=False):
    new_array = np.zeros(old_array.shape)
    for row_idx, row in enumerate(old_array):
        for elm_idx, elm in enumerate(row):
            new_array[row_idx][elm_idx] = (elm - min) * 255.0 / float(max - min)
    if reverse:
        new_array = 255 - new_array
    return new_array

def plot_overview(depth_array, grid, counter, image_per_row):
    sub = plt.subplot(grid[counter / image_per_row, counter % image_per_row])
    sub.axes.get_xaxis().set_visible(False)
    sub.axes.get_yaxis().set_visible(False)
    sub.imshow(depth_array, cmap=cm.gray)


def crop(depth_array, max_size):
    _, threshold = cv2.threshold(depth_array.copy().astype(np.uint8), 128, np.amax(depth_array), 1)
    points = cv2.findNonZero(threshold)
    x, y, w, h = cv2.boundingRect(points)
    print (x,y,w,h)
    # rect_side = max(h, w)
    # rect_side = min(rect_side, max_size)
    rect_side = max_size
    print (x, y, rect_side)
    y = y - 15 if y - 15 > 0 else 0
    x = x - 15 if x - 15 > 0 else 0
    depth_array = depth_array[y:y + rect_side, x:x + rect_side]
    return depth_array


def substract_background(depth_array, conf_array, empty_pixel_val=254):
    # median filter and gaussian filter
    depth_array = med_filter(depth_array, 3)
    depth_array = gaussian_filter(depth_array, 0.5)

    # remove low confidence and high dist pixels
    low_conf_ind = conf_array < np.median(conf_array) * 1.1
    high_dep_ind = depth_array > np.median(depth_array) * 0.9
    depth_array[low_conf_ind] = empty_pixel_val
    depth_array[high_dep_ind] = empty_pixel_val

    # smooth again
    depth_array = gaussian_filter(depth_array, 0.25)
    depth_array = med_filter(depth_array, 5)
    return depth_array

def build_confidence(depth_array, conf_array, empty_pixel_val=1):
    new_array = np.zeros(depth_array.shape)
    for row_idx, row in enumerate(depth_array):
        for elm_idx, elm in enumerate(row):
            if conf_array[row_idx][elm_idx] > 0:
                new_array[row_idx][elm_idx] = elm * conf_array[row_idx][elm_idx]
            else:
                new_array[row_idx][elm_idx] = empty_pixel_val
    return new_array

def preprocess_images(rootdir, image_per_row):
    global confi_path
    np.set_printoptions(threshold='nan')

    grid = gridspec.GridSpec(image_per_row, image_per_row, wspace=0.0, hspace=0.0)
    for subdir, dirs, files in os.walk(rootdir):
        counter = 0
        for file in files:
            if counter > image_per_row * image_per_row - 1:
                continue
            image_path = os.path.join(subdir, file)
            if ("confi" in image_path or not image_path.endswith('.png')):
                continue
            if ("depth" in image_path):
                confi_path = image_path.replace("depth_", "confi_")

            print "Reading", image_path

            im = Image.open(image_path)
            conf = Image.open(confi_path)
            depth_array = np.array(im)
            confarray = np.array(conf)

            depth_array = substract_background(depth_array, confarray)
            depth_array = crop(depth_array, 150)

            plot_overview(depth_array, grid, counter, image_per_row)

            counter += 1

    plt.axis('off')
    plt.show()

def preprocess_image(image_path, output_path):
    confi_path = image_path.replace("depth_", "confi_")

    im = Image.open(image_path)
    conf = Image.open(confi_path)
    depth_array = np.array(im)
    conf_array = np.array(conf)

    depth_array = substract_background(depth_array, conf_array, empty_pixel_val=np.amin(depth_array))
    depth_array = build_confidence(depth_array, conf_array, empty_pixel_val=np.amin(depth_array))

    #depth_array = normalize_to_gray_scale(depth_array, np.amin(depth_array), np.amax(depth_array), reverse=True)
    depth_array = crop(depth_array, 150)
    print "save", output_path
    scipy.misc.imsave(output_path, depth_array)

def walk_data_folder(rootdir, output_dir="processed/"):
    for subject_folder in os.listdir(rootdir):
        subject_dir = os.path.join(rootdir, subject_folder)
        if os.path.isdir(subject_dir):
            walk_subject_folder(rootdir, subject_folder, output_dir)

def walk_subject_folder(rootdir, subject_folder, output_dir="processed/"):
    subject_dir = os.path.join(rootdir, subject_folder)
    os.makedirs(os.path.join(output_dir, subject_folder))
    for gesture_folder in os.listdir(subject_dir):
        gesture_dir = os.path.join(rootdir, subject_folder, gesture_folder)
        if os.path.isdir(gesture_dir):
            walk_gesture_folder(rootdir, subject_folder, gesture_folder, output_dir="processed/")

def walk_gesture_folder(rootdir, subject_folder, gesture_folder, output_dir="processed/"):
    gesture_dir = os.path.join(rootdir, subject_folder, gesture_folder)
    os.makedirs(os.path.join(output_dir, subject_folder, gesture_folder))

    print "Reading folder", gesture_dir
    results_pool = []
    pool = Pool(processes=40)

    for image_file in os.listdir(gesture_dir):
        image_path = os.path.join(gesture_dir, image_file)
        if os.path.isfile(image_path):
            if "confi" in image_path or not image_path.endswith(".png"):
                continue
            image_output_path = os.path.join(output_dir, subject_folder, gesture_folder, image_file)
            results_pool.append(pool.apply_async(preprocess_image, (image_path,image_output_path,)))
            # preprocess_image(image_path, image_output_path)

    pool.close()
    pool.join()

if __name__ == "__main__":

    from time import gmtime, strftime
    date = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    rootdir = 'SSF/'
    outputdir = 'processed'

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    else:
        print "Remove output folder before making a new one :)"
        exit()

    if (len(sys.argv) < 2):
        print "Processing all images."
        walk_data_folder(rootdir)

    if (len(sys.argv) == 2):
        print "Processing images for subject: ", sys.argv[1]
        subject_folder = 'ssf14-' + sys.argv[1] + '-depth/'
        walk_subject_folder(rootdir, subject_folder)

    if (len(sys.argv) == 3):
        print "processing images for subject:", sys.argv[1], "gesture:", sys.argv[2]
        subject_folder = 'ssf14-' + sys.argv[1] + '-depth/'
        walk_gesture_folder(rootdir, subject_folder, sys.argv[2])

    print "Now:", date
    print "Rename output to:", outputdir+'_'+date
    os.rename(outputdir, outputdir+'_'+date)
