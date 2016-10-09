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
	new_array =  np.zeros(old_array.shape)
	for row_idx, row in enumerate(old_array):
		for elm_idx, elm in enumerate(row):
			new_array[row_idx][elm_idx] = (elm - min) * 255 / float(max - min)
	return new_array

def plot(imarray, grid, counter, image_per_row):
	sub = plt.subplot(grid[counter/image_per_row, counter%image_per_row])
	sub.axes.get_xaxis().set_visible(False)
	sub.axes.get_yaxis().set_visible(False)
	sub.imshow(imarray, cmap=cm.gray)

def preprocess_images(rootdir, image_per_row):
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

			imarray = np.array(im)
			confarray=np.array(conf)

			# median filter and gaussian filter
			imarray = med_filter(imarray, 3)
			imarray = gaussian_filter(imarray, 0.5)

			#remove low confidence and high dist pixels
			max_dist = np.amax(imarray);
			low_conf_ind =  confarray < np.median(confarray) * 1.15
			high_dep_ind = imarray > np.median(imarray) * 0.85
			imarray[low_conf_ind] = 1
			imarray[high_dep_ind] = 1
			imarray = normalize_to_gray_scale(imarray, np.amin(imarray), max_dist)
			
			imarray = gaussian_filter(imarray, 0.25)
			imarray = med_filter(imarray, 5)

			# bounding box
			imarray = crop(imarray, 150)
			plot(imarray, grid, counter, image_per_row)

			counter+= 1

	# with open('array_test.txt', 'w') as f:
	# 	f.write(imarray)
	plt.axis('off')
	plt.show()

def crop(imarray, max_size):
	_, threshold = cv2.threshold(imarray.astype(np.uint8), 1, np.amax(imarray), 0)
	points = cv2.findNonZero(threshold)
	x,y,w,h = cv2.boundingRect(points)
	# print (x,y,w,h)
	rect_side = max(h,w)
	rect_side = min(rect_side, max_size)
	# print (x, y, rect_side)
	imarray = imarray[y-10:y+rect_side-10, x-10:x+rect_side-10]
	return imarray




if __name__ == "__main__":
	rootdir = 'SSF/ssf14-{subject}-depth/{gesture}'
	subject = sys.argv[1]
	gesture_id = sys.argv[2]
	image_per_row = sys.argv[3]

	rootdir = rootdir.replace('{subject}', subject)
	rootdir = rootdir.replace('{gesture}', gesture_id)
	preprocess_images(rootdir, int(image_per_row))

