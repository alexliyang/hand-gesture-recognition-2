import numpy as np
import os, math, time, argparse
from PIL import Image
import scipy.misc
import skimage.transform
from FeatureExtractor import extract_feature
from sklearn.externals import joblib
from multiprocessing import Pool

bdt = joblib.load('ada100.pkl')
normalised_size = (100, 100)

def merge_slices(img_file_list):
	x1, x2, y1, y2 = [],[],[],[]
	for img_file_path in img_file_list:
		order = img_file_path[img_file_path.rfind('_'):-4]		
		text_file_path = img_file_path[:-4] + '.txt'
		with open(text_file_path, 'r') as text_file:
			coordinates = text_file.read().strip().split(',')
			x1.append(int(coordinates[0]))
			x2.append(int(coordinates[1]))
			y1.append(int(coordinates[2]))
			y2.append(int(coordinates[3]))

	x1 = min(x1)
	x2 = max(x2)
	y1 = min(y1)
	y2 = max(y2)
	# print "coordinate_dict of merged coordinate:", x1, x2, y1, y2
	return (x1, y1, x2, y2)

def classify_image(file_path):
	# print "classify:", file_path
	depth_array = np.array(Image.open(file_path))
	features = extract_feature(depth_array)
	class_name = bdt.predict([features])[0]
	if class_name == 'T':
		# print file_path, class_name
		return file_path
	else:
		return ''

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--input', default="cropped_merged")
	parser.add_argument('--output', default="cropped_final")
	parser.add_argument('--original', default="processed_merged")
	args = parser.parse_args()

	input_dir = args.input
	output_dir = args.output
	original_dir = args.original

	for gesture_folder in os.listdir(input_dir):
		gesture_dir = os.path.join(input_dir, gesture_folder)
		if os.path.isdir(gesture_dir):
			temp_file_dict = {}
			print "Filter cropped images for gesture", gesture_folder
			start = time.time()
			# pool = Pool(processes=40)
			for file_name in os.listdir(gesture_dir):
				if file_name.endswith('.png'):
					prefix = file_name[:file_name.find('.png')]

					if not prefix in temp_file_dict:
						temp_file_dict[prefix] = []

					file_path = os.path.join(gesture_dir, file_name)
					temp_file_dict[prefix].append(classify_image(file_path))
					# temp_file_dict[prefix].append(pool.apply_async(classify_image, (file_path), ))
					# classify_image(file_path, temp_file_dict)

			# pool.close()
			# pool.join()

			# print "Pool joined", len(temp_file_dict)

			for prefix in temp_file_dict:
				temp_file_dict[prefix] = [x for x in temp_file_dict[prefix] if len(x) > 1]

			print "Finished processing in", time.time() - start,"\nSlicing original..."

			output_image_dir = os.path.join(output_dir, gesture_folder)
			original_gesture_dir = os.path.join(original_dir, gesture_folder)
			if not os.path.exists(output_image_dir):
				os.makedirs(output_image_dir)
			count = 0
			for key in temp_file_dict.keys():
				if len(temp_file_dict[key]) > 0:
					count += 1
					x1, y1, x2, y2 = merge_slices(temp_file_dict[key])
					original = np.array(Image.open(os.path.join(original_gesture_dir, key + '.png')))
					sliced = original[x1:x2, y1:y2]
					sliced = skimage.transform.resize(sliced, normalised_size, preserve_range=True)

					img = scipy.misc.toimage(sliced, high=np.amax(sliced), low=np.amin(sliced), mode='I')
					img.save(os.path.join(output_image_dir, '{}.png'.format(key)))
			print count, "images survived out of", len(temp_file_dict)


