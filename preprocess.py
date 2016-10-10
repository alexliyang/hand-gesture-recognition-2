from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from scipy.signal import medfilt as med_filter
from scipy.ndimage.filters import gaussian_filter
import cv2, os, sys

sys.path.append('/usr/local/lib/python2.7/site-packages')


def normalize_to_gray_scale(old_array, min, max):
    new_array = np.zeros(old_array.shape)
    for row_idx, row in enumerate(old_array):
        for elm_idx, elm in enumerate(row):
            new_array[row_idx][elm_idx] = (elm - min) * 255 / float(max - min)
    return new_array


def plot_overview(depth_array, grid, counter, image_per_row):
    """
	:param deptharray:
	:param grid:
	:param counter:
	:param image_per_row:
	"""
    sub = plt.subplot(grid[counter / image_per_row, counter % image_per_row])
    sub.axes.get_xaxis().set_visible(False)
    sub.axes.get_yaxis().set_visible(False)
    sub.imshow(depth_array, cmap=cm.gray)


def crop(depth_array, max_size):
    _, threshold = cv2.threshold(depth_array.astype(np.uint8), 1, np.amax(depth_array), 0)
    points = cv2.findNonZero(threshold)
    x, y, w, h = cv2.boundingRect(points)
    # print (x,y,w,h)
    rect_side = max(h, w)
    rect_side = min(rect_side, max_size)
    # print (x, y, rect_side)
    depth_array = depth_array[y - 10:y + rect_side - 10, x - 10:x + rect_side - 10]
    return depth_array


def substract_background(depth_array, confarray, empty_pixel_val=1):
    # median filter and gaussian filter
    depth_array = med_filter(depth_array, 3)
    depth_array = gaussian_filter(depth_array, 0.5)

    # remove low confidence and high dist pixels
    max_dist = np.amax(depth_array);
    low_conf_ind = confarray < np.median(confarray) * 1.15
    high_dep_ind = depth_array > np.median(depth_array) * 0.85
    depth_array[low_conf_ind] = empty_pixel_val
    depth_array[high_dep_ind] = empty_pixel_val
    depth_array = normalize_to_gray_scale(depth_array, np.amin(depth_array), max_dist)

    # smooth again
    depth_array = gaussian_filter(depth_array, 0.25)
    depth_array = med_filter(depth_array, 5)
    return depth_array


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


if __name__ == "__main__":
    rootdir = 'SSF/ssf14-{subject}-depth/{gesture}'
    subject = sys.argv[1]
    gesture_id = sys.argv[2]
    image_per_row = sys.argv[3]

    rootdir = rootdir.replace('{subject}', subject)
    rootdir = rootdir.replace('{gesture}', gesture_id)
    preprocess_images(rootdir, int(image_per_row))
