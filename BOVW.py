import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import sklearn.utils

from glob import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support


class BOVW:

	def __init__(self, estimator, n_bags=200, max_features_per_image=200,
	             edge_threshold=50, contrast_threshold=0.02, tol=0.001,
	             is_resample=False, resample_factor=3.0, probability=False,
	             verbose=False, random_state=None, color="HSV",
	             is_reuse=False, class_threshold=None):

		# Default variables
		self.estimator = estimator
		self.n_bags = n_bags
		self.max_features_per_image = max_features_per_image
		self.edge_threshold = edge_threshold
		self.contrast_threshold = contrast_threshold
		self.tol = tol
		self.is_resample = is_resample
		self.resample_factor = resample_factor
		self.probability = probability
		self.verbose = verbose
		if random_state is None:
			self.random_state = int((1234.0 - 2.0) * np.random.random_sample() + 2.0)
		else:
			self.random_state = int(random_state)
		if color is "HSV":
			self.color = cv2.COLOR_RGB2HSV
		elif color is "HLS":
			self.color = cv2.COLOR_RGB2HLS
		elif color is "YUV":
			self.color = cv2.COLOR_RGB2YUV
		elif color is "GRAY":
			self.color = cv2.COLOR_RGB2GRAY
		else:
			self.color = None
		self.is_reuse = is_reuse
		self.class_threshold = class_threshold

		# protecte variables
		self._scale = StandardScaler()
		self._xfeat = cv2.xfeatures2d.SIFT_create(self.max_features_per_image,
		                                          edgeThreshold=self.edge_threshold,
		                                          contrastThreshold=self.contrast_threshold)
		if self.is_reuse:
			self._kmeans = pickle.load(open("models/kmeans.sav", 'rb'))
		else:
			self._kmeans = KMeans(n_clusters=self.n_bags,
			                      tol=self.tol,
			                      random_state=self.random_state)
		self._labels_dict = {}  # label to encoded label
		self._histogram = None

	def fit(self, train_path):
		"""Train images in train_path"""

		if self.verbose:
			print("Fetch train images....")

		images_dict, n_images = self.__get_files(train_path, True)

		labels = np.array([])
		features = []
		encoded = 0
		for label, images in images_dict.items():
			self._labels_dict[str(encoded)] = label

			if self.verbose:
				print("Computing features for", label)

			for image in images:
				labels = np.append(labels, encoded)
				kp, des = self.__extract_features(image)
				features.append(des)

			encoded += 1

		features_v = self.__formart_features(features)

		if self.verbose:
			print("Clustering...")

		if self.is_reuse:
			kmeans_bags = self._kmeans.predict(features_v)
		else:
			kmeans_bags = self._kmeans.fit_predict(features_v)

			if self.verbose:
				pickle.dump(self._kmeans, open("models/kmeans.sav", 'wb'))

		self._histogram = np.array([np.zeros(self.n_bags) for i in range(n_images)])
		count = 0
		for i in range(n_images):
			if features[i] is not None:
				ll = len(features[i])
				for j in range(ll):
					idx = kmeans_bags[count + j]
					self._histogram[i][idx] += 1
				count += ll

		if self.verbose:
			print("Vocabulary Histogram Generated")
			print("Normalize...")

		self._histogram = self._scale.fit_transform(self._histogram)

		if self.verbose:
			print("Histogram shape", self._histogram.shape)
			print("Trainning...")

		self.estimator.fit(self._histogram, labels)

		if self.verbose:
			print(self.estimator)
			pickle.dump(self.estimator, open("models/estimator.sav", 'wb'))
			print("Trainning done!")

	def fit_score(self, train_path, test_path):
		self.fit(train_path)
		self.score(test_path)

	def predict(self, image):
		"""Predict a single image"""

		kp, des = self.__extract_features(image)
		name, final_predict, raw_predict = None, None, None
		if len(kp) != 0:
			vocab = np.array([0 for i in range(self.n_bags)])
			test_res = self._kmeans.predict(des)

			for each in test_res:
				vocab[each] += 1

			vocab = self._scale.transform(vocab.reshape(1, -1) / 1.0)

			if self.probability:
				raw_predict = self.estimator.predict_proba(vocab)
				if self.class_threshold is None:
					final_predict = np.argmax(raw_predict[0])
				else:
					final_predict = np.argmax(raw_predict[0] - self.class_threshold)
			else:
				final_predict = self.estimator.predict(vocab)[0]

			name = self._labels_dict[str(int(final_predict))]

		return name, final_predict, raw_predict

	def score(self, test_path):
		"""Test images in train_path"""

		if self.verbose:
			print("Fetch test images...")

		images_dict, n_images = self.__get_files(test_path)

		true, count = 0.0, 0.0
		for label, images in images_dict.items():
			if self.verbose:
				print("Processing", label)

			for img in images:
				name, target, raw_predict = self.predict(img)

				if name == label:
					if self.verbose:
						if self.probability:
							print("Predict", label, name, raw_predict, "TRUE")
						else:
							print("Predict", label, name, target, "TRUE")
					true += 1.0
				elif self.verbose:
					if self.probability:
						print("Predict", label, name, raw_predict)
					else:
						print("Predict", label, name, target)

				count += 1.0

		return true / count

	def score_percision_recall(self, test_path):
		if self.verbose:
			print("Fetch test images...")

		images_dict, n_images = self.__get_files(test_path)

		y_true, y_pred = [], []
		for label, images in images_dict.items():
			if self.verbose:
				print("Processing", label)

			for img in images:
				name, target, raw_predict = self.predict(img)

				y_true.append(label)
				y_pred.append(name)

		y_true = np.asarray(y_true)
		y_pred = np.asarray(y_pred)

		percision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average="micro")

		return percision, recall

	def plot_histogram(self, vocab=None):
		"""Draw histogram"""

		if self.verbose:
			print("Plotting histogram...")

		if vocab is None:
			vocab = self._histogram

		x = np.arange(self.n_bags)
		y = np.array([abs(np.sum(vocab[:, h], dtype=np.int32)) for h in range(self.n_bags)])

		plt.bar(x, y)
		plt.xlabel("Visual Word Index")
		plt.ylabel("Frequency")
		plt.title("Complete Vocabulary Generated")
		plt.xticks(x + 0.4, x)
		plt.show()

	def __formart_features(self, features):
		"""Format features to vertical"""

		v_stack = np.array(features[0])
		for remaining in features:
			if remaining is not None:
				v_stack = np.vstack((v_stack, remaining))
		features = v_stack.copy()
		if self.verbose:
			print("Features shape", features.shape)
		return features

	def __extract_features(self, image):
		"""Extract features of images"""

		image_temp = image
		if self.color is not None:
			image_temp = cv2.cvtColor(image, self.color)

		kp = self._xfeat.detect(image_temp, None)
		return self._xfeat.compute(image, kp)

	def __get_files(self, path, is_balance=False):
		"""Get images dictionary from path
			Keys is label
			Values is list of images
		"""

		images = {}
		n_sample, count = 0, 0
		for each in glob(path + "*"):
			# Extract lable from path
			label = each.split("\\")[-1]
			if self.verbose:
				print("Reading image from", label)

			images[label] = []
			images_list = glob(path + label + "/*")
			k = len(images_list)
			if n_sample == 0:
				n_sample = k
			if n_sample > k:
				n_sample = int(np.ceil((n_sample + k) / self.resample_factor))
			for image in images_list:
				img = cv2.imread(image)
				images[label].append(img)
				count += 1

			# Shuffle
			images[label] = sklearn.utils.shuffle(images[label], random_state=self.random_state)

		if is_balance:
			import random as rand
			rand.seed = self.random_state

			count = 0
			for label in images.keys():
				k = len(images[label])
				if n_sample < k:
					k = n_sample
				images[label] = rand.sample(images[label], k)
				print("Sampling category", label, k, " images")
				count += k

		if self.verbose:
			print("Fetched", count, "images")

		return images, count

	def __resample(self, images_dict):
		"""Resample dataset to prevent imbalanced data"""

		# Get class have max images
		n = np.mean([len(images_dict[key]) for key in images_dict.keys()])
		count = 0
		for key in images_dict.keys():
			k = len(images_dict[key])
			if k > n:
				k = int(np.ceil((k + n) / 10))
				# Resample some images
				images_dict[key] = sklearn.utils.resample(images_dict[key], n_samples=k, random_state=self.random_state)

			if self.verbose:
				print("Resample", key, ":", k, "images")
			count += k

		return images_dict, count
