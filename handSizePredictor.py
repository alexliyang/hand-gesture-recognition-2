import numpy as np


class HandSizePredictor:
	def __init__(self, degree=2):
		self.degree = degree

	def train(self, tranining_file):
		print "Training detector with:", tranining_file		
		# x - depth of hand
		x = []
		# y - desired hand size (radius)
		y = []
		with open(tranining_file) as training_pairs:
			for line in training_pairs:
				pairs = line.strip().split('\t')
				x.append(float(pairs[1]))
				y.append(float(pairs[0]))

		x = np.array(x)
		y = np.array(y)
		# trained params
		self.params = np.polyfit(x, y, 2)
		# init predictor
		self.predictor = np.poly1d(self.params)
		print "Finished. Params:", self.predictor

	def predict(self, x):
		return self.predictor(x)

if __name__ == "__main__":
	hsp = HandSizePredictor()
	hsp.train('hand_size.txt')



