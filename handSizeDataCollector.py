import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from handSizeDetector import HandSizePredictor

original_directory = 'processed_full_subtract_only'
labeled_directory = 'hand_size_labeled'
hand_size_training_file = open('hand_size.txt', 'w')
output_string = ""
max_ave = 0
for subject_folder in os.listdir(labeled_directory):
	if os.path.isdir(os.path.join(labeled_directory, subject_folder)):
		for gesture_folder in os.listdir(os.path.join(labeled_directory, subject_folder)):
			color=np.random.rand(50)
			if os.path.isdir(os.path.join(labeled_directory, subject_folder, gesture_folder)):
				for img in os.listdir(os.path.join(labeled_directory, subject_folder, gesture_folder)):
					if img.endswith('.png'):
						im = Image.open(os.path.join(labeled_directory, subject_folder, gesture_folder, img))
						depth_array = np.array(im)
						im_original = Image.open(os.path.join(original_directory, subject_folder, gesture_folder, img))
						depth_array_original = np.array(im_original)

						rect = np.where(depth_array < 1)
						y = np.amin(rect[0]) + 1
						h = np.amax(rect[0]) - np.amin(rect[0]) - 2
						x = np.amin(rect[1]) + 1
						w = np.amax(rect[1]) - np.amin(rect[1]) - 2

						if w < 5 or h < 5:
							continue

						# print "For image:", os.path.join(labeled_directory, subject_folder, gesture_folder, img)

						cropped_original = depth_array_original[y:y + h, x:x + w]
						cropped_original_hand_mask = cropped_original > 19999
						masked_cropped_depth_array = np.ma.array(cropped_original, mask=cropped_original_hand_mask)
						average_depth = np.ma.sum(masked_cropped_depth_array)/(w * h)

						if average_depth > max_ave:
							max_ave = average_depth

						output_string += str(w) + '\t' + str(average_depth) + '\n'
						plt.scatter(average_depth, w, color=color)

hand_size_training_file.write(output_string)
hand_size_training_file.close()

hsp = HandSizePredictor()
hsp.train('hand_size.txt')
depth_list = []
width_list = []
for depth in range(30, max_ave):
	depth_list.append(depth)
	width_list.append(hsp.predict(depth))

plt.plot(depth_list, width_list, 'r')
plt.show()
print "done."
