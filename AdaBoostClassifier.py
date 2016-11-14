import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, time
import pickle as pk
from sklearn.externals import joblib
from FeatureExtractor import extract_feature


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

input_folder = "cropped_merged_100"

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)

def construct_label_dict(text_file):
	f = open(text_file, 'r')
	lines = f.read().split('\n')
	img_dict = {}
	for line in lines:
		kv_pair = line.split('\t')
		if len(kv_pair) < 2: continue;
		img_dict[kv_pair[0]] = kv_pair[1]
	return img_dict

def construct_feature_array(img_file):
	# start = time.time()
	depth_array = np.array(Image.open(img_file))
	# print time.time() - start, img_file
	return extract_feature(depth_array)

#generate training data

input_folder = "cropped_merged_100"

features = []
labels = []

print "Preparing training data..."

for gesture_folder in os.listdir(input_folder):
	gesture_dir = os.path.join(input_folder, gesture_folder)
	if os.path.isdir(gesture_dir):
		print "Processing", gesture_folder
		img_dict = construct_label_dict(os.path.join(gesture_dir, 'class.txt'))

		for img in os.listdir(gesture_dir):
			if img.endswith('.png'):
				fv = construct_feature_array(os.path.join(gesture_dir, img))
				label = img_dict[img]
				
				features.append(fv)
				labels.append(label)

print len(features), "data points processed.\nTraining..."

start = time.time()
bdt.fit(features, labels)

print "Training finished in", time.time() - start

joblib.dump(bdt, 'ada100.pkl') 

print "model dumped."

#test
test_folder = 'test_60/'
p, n, fp, fn = 0, 0, 0, 0
for img in os.listdir(test_folder):
	test_img_dict = construct_label_dict(os.path.join(test_folder, 'class.txt'))

	if img.endswith('.png'):
		fv = construct_feature_array(os.path.join(test_folder, img))
		label = bdt.predict([fv])
		if label[0] == 'T':
			p += 1
		else:
			n += 1
		if test_img_dict[img] == 'T' and label[0] == 'F':
			fn += 1
		if test_img_dict[img] == 'F' and label[0] == 'T':
			fp += 1
		# print 'class:', test_img_dict[img],'predicted:', label[0], '-', img

print 'p =',p,'n=',n, 'fp =', fp/60.0, '; fn =', fn/60.0


