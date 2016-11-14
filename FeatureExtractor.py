import numpy as np
import scipy.misc
import math


# build feature map

# build feature for each input img

def extract_feature(depth_array):
	size = depth_array.shape[0]
	feature_size_step = 1
	max_nx, max_ny = 10, 10
	feature_pool = []
	for nx in range(1, max_nx+1, feature_size_step):
		for ny in range(1, max_ny+1, feature_size_step):
			stepx = size / (2 * nx)
			stepy = size / (2 * ny)
			blockw = size / nx
			blockh = size / ny
			# print "For nx:", nx, "ny:", ny, "step:", stepx, stepy, 'block:', blockw, blockh

			for x in range(0, 2 * nx - 1):
				for y in range(0, 2 * ny - 1):
					# print x,y, '-',
					# print y * stepy, y*stepy + blockh, x * stepx, x * stepx + blockw
					roi = depth_array[y * stepy: y*stepy + blockh, x * stepx: x * stepx + blockw]
					# print "roi:", roi.shape
					depth_center = roi[blockh/2][blockw/2]
					area = np.sum(roi)/(1.0 * blockw * blockh)
					feature_pool.append(depth_center - area)

	return feature_pool

if __name__ == "__main__":
	img = scipy.misc.imread('hand_cropped/ssf14-0-depth/1/depth_17.png_0.png')

	depth_array = np.array(img)

	f = extract_feature(depth_array)
	print len(f), f[:20]






