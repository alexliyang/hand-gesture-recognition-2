from handSizePredictor import HandSizePredictor
from PIL import Image
import numpy as np
import scipy.misc
import skimage.transform
import os, argparse

hsp = HandSizePredictor()
hsp.train('hand_size.txt')

def generate_possible_cuts(input_img_path, output_img_path, step=20, w=319, h=239, x=0, y=0):
	depth_array = np.array(Image.open(input_img_path))

	output_counter = 0

	normalised_size = (100, 100)
	while y + step < h:
		y += step
		x = 0 
		while x + step < w:
			x += step
			# print x-step, x+step, y-step, y+step,
			img_slice = depth_array[y-step:y+step,x-step:x+step]
			ave_depth = np.sum(img_slice) / (4 * step * step)
			# print ave_depth

			if ave_depth > 5000:
				continue

			# predicted_size = hsp.predict(ave_depth)
			predicted_size = int(hsp.predict(depth_array[y][x]))

			if predicted_size < step or predicted_size > min(w, h):
				continue	

			if predicted_size % 2 == 1:
				predicted_size += 1

			if x - predicted_size/2 < 0 or x + predicted_size > w or y - predicted_size/2 < 0 or y + predicted_size/2 > h:
				continue;

			predicted_slice = depth_array[ max(y-predicted_size/2,0):min(y+predicted_size/2, h), max(x-predicted_size/2, 0):min(x+predicted_size/2, w)]			

			predicted_slice = skimage.transform.resize(predicted_slice, normalised_size, preserve_range=True)

			img = scipy.misc.toimage(predicted_slice, high=np.amax(predicted_slice), low=np.amin(predicted_slice), mode='I')
			img.save(output_img_path + '_' + str(output_counter) + '.png')
			f = open(output_img_path + '_' + str(output_counter) + '.txt', 'w')
			f.write("{},{},{},{}".format(max(y-predicted_size/2,0), min(y+predicted_size/2, h), max(x-predicted_size/2, 0), min(x+predicted_size/2, w)))
			f.close()
			output_counter += 1

	if output_counter < 1:
		print input_img_path, "failed at cropping."

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--input', default="processed_merged")
	parser.add_argument('--output', default="cropped_merged")
	parser.add_argument('--step', type=int, default=20)
	args = parser.parse_args()
	
	input_folder = args.input
	output_folder = args.output
	step = args.step

	for gesture_folder in os.listdir(input_folder):
		if os.path.isdir(os.path.join(input_folder, gesture_folder)):
			output_image_folder = os.path.join(output_folder, gesture_folder)
			if not os.path.exists(output_image_folder):
				os.makedirs(output_image_folder)

			print "Cropping images for:", gesture_folder

			for img in os.listdir(os.path.join(input_folder, gesture_folder)):
				if img.endswith('.png'):
					input_img_path = os.path.join(input_folder, gesture_folder, img)
					output_img_path = os.path.join(output_image_folder, img)
					generate_possible_cuts(input_img_path, output_img_path, step=step)
			with open(os.path.join(output_image_folder, 'class.txt'), 'w') as f:
				output_string = ""
				for img in os.listdir(output_image_folder):
					if img.endswith('png'):
						output_string += img + '\t\n'
				f.write(output_string)



