import os, random,shutil
input_folder = "cropped_merged"
output_folder = "cropped_merged_100"

for gesture_folder in os.listdir(input_folder):
	if os.path.isdir(os.path.join(input_folder, gesture_folder)):
		output_image_folder = os.path.join(output_folder, gesture_folder)
		if not os.path.exists(output_image_folder):
			os.makedirs(output_image_folder)

		print "Cropping images for:", gesture_folder

		f = open(os.path.join(output_image_folder, 'class.txt'), 'w')
		output_string = ""
		png_files = [ img for img in os.listdir(os.path.join(input_folder, gesture_folder)) if img.endswith('.png')]
		for x in range(0, 100):
			img = random.choice(png_files)
			img_src = os.path.join(input_folder, gesture_folder, img)
			print img_src, os.path.join(output_image_folder, gesture_folder  + str(x)+'.png')
			shutil.copy(img_src, os.path.join(output_image_folder, gesture_folder  + str(x)+'.png'))
			output_string += gesture_folder + str(x) + '.png\t\n'
		f.write(output_string)
		f.close()