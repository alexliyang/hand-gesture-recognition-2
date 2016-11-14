import shutil, os, argparse



if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--input', default="processed_full_subtract_only")
	parser.add_argument('--output', default="processed_merged")
	args = parser.parse_args()
	
	input_folder = args.input
	output_folder = args.output

	for subject_folder in os.listdir(input_folder):
		if os.path.isdir(os.path.join(input_folder, subject_folder)):

			for gesture_folder in os.listdir(os.path.join(input_folder, subject_folder)):
				if os.path.isdir(os.path.join(input_folder, subject_folder, gesture_folder)):

					output_image_folder = os.path.join(output_folder, gesture_folder)

					if not os.path.exists(output_image_folder):
						os.makedirs(output_image_folder)

					print "Merge images in:", subject_folder, gesture_folder

					for img in os.listdir(os.path.join(input_folder, subject_folder, gesture_folder)):
						if img.endswith('.png'):
							img_name = subject_folder + '_' + gesture_folder + '_' + img

							shutil.copy(os.path.join(input_folder, subject_folder, gesture_folder, img), os.path.join(output_image_folder, img_name))