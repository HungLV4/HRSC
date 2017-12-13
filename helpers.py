import cv2
import numpy as np
import random as rand
import pickle
import math
from glob import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from skimage.feature import hog


class ImageHelpers:
	def __init__(self):
		self.sift_object = cv2.xfeatures2d.SIFT_create(200, edgeThreshold=50, contrastThreshold=0.02)
		# self.sift_object = cv2.xfeatures2d.SURF_create(extended=True)
		# self.sift_object = cv2.ORB_create(patchSize=32, edgeThreshold=15, nlevels=4)    # 66%

	def trans(self, image):
		return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

	def features(self, image):
		kp = self.sift_object.detect(self.trans(image), None)
		kp, des = self.sift_object.compute(image, kp)
		# keypoints, descriptors = self.sift_object.detectAndCompute(self.trans(image), None)
		# print(len(keypoints))
		return [kp, des]


class BOVHelpers:
	def __init__(self, n_clusters=20):
		self.n_clusters = n_clusters
		# self.kmeans_obj = pickle.load(open("models/model_Kmeans.sav", 'rb'))
		self.kmeans_obj = KMeans(n_clusters=n_clusters, tol=0.1, random_state=111)
		self.kmeans_ret = None
		self.descriptor_vstack = None
		self.mega_histogram = None
		self.scale = StandardScaler()
		# self.clf = pickle.load(open("models/model_Classifier.sav", 'rb'))
		self.clf = SVC(kernel="rbf", C=2.8, gamma=0.0073, cache_size=10000, probability=False, random_state=111)
		# self.clf = RandomForestClassifier(n_estimators=90, max_depth=10, random_state=111)
		# self.clf = GradientBoostingClassifier(n_estimators=90, max_depth=5, subsample=0.9)
		# self.clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=150, n_jobs=-1, multi_class="multinomial")
		# self.clf = AdaBoostClassifier(n_estimators=60, random_state=111)

	def cluster(self):
		"""	
		cluster using KMeans algorithm, 

		"""
		print("Features:", self.descriptor_vstack.shape)
		# self.kmeans_ret = self.kmeans_obj.predict(self.descriptor_vstack)
		self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)
		pickle.dump(self.kmeans_obj, open("models/model_Kmeans.sav", 'wb'))

	def developVocabulary(self, n_images, descriptor_list, kmeans_ret=None):

		"""
		Each cluster denotes a particular visual word 
		Every image can be represeted as a combination of multiple 
		visual words. The best method is to generate a sparse histogram
		that contains the frequency of occurence of each visual word 

		Thus the vocabulary comprises of a set of histograms of encompassing
		all descriptions for all images

		"""

		self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
		old_count = 0
		for i in range(n_images):
			if descriptor_list[i] is not None:
				l = len(descriptor_list[i])
				for j in range(l):
					if kmeans_ret is None:
						idx = self.kmeans_ret[old_count + j]
					else:
						idx = kmeans_ret[old_count + j]
					self.mega_histogram[i][idx] += 1
				old_count += l
		print("Vocabulary Histogram Generated")

	def standardize(self, std=None):
		"""
		
		standardize is required to normalize the distribution
		wrt sample size and features. If not normalized, the classifier may become
		biased due to steep variances.

		"""
		if std is None:
			self.scale.fit(self.mega_histogram)
			self.mega_histogram = self.scale.transform(self.mega_histogram)
		else:
			print("STD not none. External STD supplied")
			self.mega_histogram = std.transform(self.mega_histogram)

	def formatND(self, l):
		"""	
		restructures list into vstack array of shape
		M samples x N features for sklearn

		"""
		vStack = np.array(l[0])
		for remaining in l:
			if remaining is not None:
				vStack = np.vstack((vStack, remaining))
		self.descriptor_vstack = vStack.copy()
		return vStack

	def train(self, train_labels):
		"""
		uses sklearn.svm.SVC classifier (SVM) 


		"""
		print(self.clf)
		# print("Train labels", train_labels)
		print("Features", self.mega_histogram.shape)
		self.clf.fit(self.mega_histogram, train_labels)
		pickle.dump(self.clf, open("models/model_Classifier.sav", 'wb'))
		print("Training completed")

	def predict(self, iplist):
		predictions = self.clf.predict(iplist)
		# predictions = self.clf.predict(iplist)
		return predictions

	def plotHist(self, vocabulary=None):
		print("Plotting histogram")
		if vocabulary is None:
			vocabulary = self.mega_histogram

		x_scalar = np.arange(self.n_clusters)
		y_scalar = np.array([abs(np.sum(vocabulary[:, h], dtype=np.int32)) for h in range(self.n_clusters)])

		print(y_scalar)

		plt.bar(x_scalar, y_scalar)
		plt.xlabel("Visual Word Index")
		plt.ylabel("Frequency")
		plt.title("Complete Vocabulary Generated")
		plt.xticks(x_scalar + 0.4, x_scalar)
		plt.show()


class FileHelpers:
	def __init__(self):
		pass

	def getFiles(self, path, isSample=False):
		"""
		- returns  a dictionary of all files 
		having key => value as  objectname => image path

		- returns total number of files.

		"""
		rand.seed = 111

		imlist = {}
		n_sample = 0
		count = 0
		for each in glob(path + "*"):
			word = each.split("\\")[-1]
			print(" #### Reading image category ", word, " ##### ")
			imlist[word] = []
			imagelist = glob(path + word + "/*")
			k = len(imagelist)
			if n_sample == 0:
				n_sample = k
			if n_sample > k:
				n_sample = math.ceil((n_sample + k) / 3)
			for imagefile in imagelist:
				# print("Reading file ", imagefile)
				im = cv2.imread(imagefile)
				imlist[word].append(im)
				count += 1
			rand.shuffle(imlist[word])

		if isSample:
			count = 0
			for word in imlist.keys():
				k = len(imlist[word])
				if n_sample < k:
					k = n_sample
				imlist[word] = rand.sample(imlist[word], k)
				print("Sampling category", word, k, " images")
				count += k

		return [imlist, count]
