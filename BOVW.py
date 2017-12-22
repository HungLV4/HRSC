import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import sklearn.utils

from glob import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support


class BOVW:
	"""Mô hình Bag Of Visual Word cho bài toán phân loại ảnh

	Các ảnh trainning được đưa vào models là các ảnh được cắt theo bounding box,
	và xoay nằm ngang, loại bỏ hoàn toàn nhiễu. Sau đó sẽ resample từ tập train
	(để khắc phục trường hợp data phân bố nghiêng).

	Các class đều là của layer 1:
		classID/
			100000002/: 168 ảnh (gồm tàu sân bay cỡ lớn và cỡ bé)
			100000003/: 686 ảnh (gồm tàu quân sự có sân bay trực thăng ở đít)
			100000004/: 413 ảnh (gồm các tàu linh tinh, đánh cá)
			100000027/: 45 ảnh (chỉ có tàu ngầm)

	Khi resample chỉ lấy 1 lượng ảnh nhất định ở các class, không lấy tất

	Parameters
    ----------
    estimator : classifier
        SVC, Logistic Regression, RandomForest

    xfeat : object phụ trách extract features từ ảnh
        Có thể là 1 trong 3 dạng sau
            - cv2.xfeatures2d.SIFT_create(250, edgeThreshold=50, contrastThreshold=0.02)
			- cv2.xfeatures2d.SURF_create(0, extended=True)
			- cv2.ORB_create(patchSize=64, edgeThreshold=10, nlevels=8)
		Thường phải thử trước

	n_bags : Số lượng bag cho mô hình BOW tương ứng với số features
		Các features được trích xuất từ ảnh sẽ được clustering bằng Kmeans vào đây
	tol : hệ số lỗi của Kmeans
		Càng lớn chạy càng nhanh nhưng có thể không hội tụ
	is_resample : boolean
		Có thực hiện resample hay không?
	resample_factor : hệ số resample
		min_data_number => số lượng data nhỏ nhất trong class
		taken_data_number = (current_data_number + min_data_number) / resample_factor
		=> lấy của class hiện tại (taken_data_number) ảnh
	probability : boolean
		Có hiển thị raw predict không?
	verbose : boolean
		Có hiển thị log hay không?
	random_state : integer
		Seed cho các hàm random
	color : "HSV", "YUV", "HLS", None
		Chuyển ảnh gốc sang màu tương ứng trước khi trích xuất features
	is_reuse : boolean
		True nếu muốn dùng lại data trainning (sau khi preprocessing) để train và thử các classifier khác nhau
	class_threshold : array()
		Đặt ngưỡng để gán label cho 1 bức ảnh khi test, sử dụng cùng raw predict
	"""

	def __init__(self, estimator, xfeat, n_bags=200, tol=0.001,
	             is_resample=False, resample_factor=3.0, probability=False,
	             verbose=False, random_state=None, color="HSV",
	             is_reuse=False, class_threshold=None):

		# Default variables
		self.estimator = estimator
		self.xfeat = xfeat
		self.n_bags = n_bags
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

		if self.is_reuse:
			try:
				# Load train data đã được tiền xử lý
				self._histogram = np.load("models/X.npy")   # features
				self._labels = np.load("models/y.npy")  # label
				self._labels_dict = np.load("models/y_dict.npy").item() # label encode
				self._kmeans = pickle.load(open("models/kmeans.sav", 'rb'))  # thông số kmeans
				self._scale = pickle.load(open("models/scale.sav", "rb"))   # thông số của scale
			except Exception as e:
				print("Can't load local file!", e)
				self.is_reuse = False

		if not self.is_reuse:
			self._histogram = None
			self._labels = np.array([])
			self._labels_dict = {}
			self._kmeans = KMeans(n_clusters=self.n_bags,
			                      tol=self.tol,
			                      random_state=self.random_state)
			self._scale = MinMaxScaler((-1, 1))  # Qua thử nghiệm thấy scale trong khoảng (-1,1) tốt hơn (0,1)

	def fit(self, train_path):
		"""Train images in train_path"""

		if not self.is_reuse:
			if self.verbose:
				print("Fetch train images....")

			images_dict, n_images = self.__get_files(train_path, True)

			features = []
			encoded = 0
			# Thực hiện encode label
			for label, images in images_dict.items():
				self._labels_dict[str(encoded)] = label

				if self.verbose:
					print("Computing features for", label)

				for image in images:
					self._labels = np.append(self._labels, encoded)
					kp, des = self.__extract_features(image)    # Lấy features
					features.append(des)

				encoded += 1

			# Sắp xếp lại các features
			features_v = self.__formart_features(features)

			if self.verbose:
				print("Clustering...")

			# Phân cụm các features => tương dương với giảm chiều
			kmeans_bags = self._kmeans.fit_predict(features_v)

			# Với mỗi ảnh, xem xem 1 ảnh với mỗi bags có bao nhiêu features thuộc vào
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

			# Scale features
			self._histogram = self._scale.fit_transform(self._histogram)

			if self.verbose:
				print("Histogram shape", self._histogram.shape)
				print("Trainning...")

		# Train classifier
		self.estimator.fit(self._histogram, self._labels)

		if self.verbose:
			print(self.estimator)
			print("Trainning done!")

	def fit_score(self, train_path):
		"""Độ chính xác trên tập train"""

		self.fit(train_path)
		return self.score(train_path)

	def predict(self, image):
		"""Predict a single image"""

		kp, des = self.__extract_features(image)
		name, final_predict, raw_predict = None, None, None
		if len(kp) != 0:
			vocab = np.array([0 for i in range(self.n_bags)])
			test_res = self._kmeans.predict(des)

			for each in test_res:
				vocab[each] += 1

			# Scale data
			vocab = self._scale.transform(vocab.reshape(1, -1) / 1.0)

			if self.probability:
				raw_predict = self.estimator.predict_proba(vocab)
				if self.class_threshold is None:
					final_predict = np.argmax(raw_predict[0])
				else:
					final_predict = np.argmax(raw_predict[0] - self.class_threshold)
			else:
				final_predict = self.estimator.predict(vocab)[0]

			# Lấy ra tên class dựa vào encode
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
							print("Predict", label, name, "TRUE")
					true += 1.0
				elif self.verbose:
					if self.probability:
						print("Predict", label, name, raw_predict)
					else:
						print("Predict", label, name)

				count += 1.0

		return true / count

	def score_percision_recall(self, test_path):
		"""Tính toán percision và recall"""

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

		percision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average=None)

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

	def persist(self):
		"""Lưu train data sau khi preprocessing, đặt is_reuse là True để sử dụng lại"""

		print("Persisting...")
		np.save("models/X.npy", self._histogram)
		np.save("models/y.npy", self._labels)
		np.save("models/y_dict.npy", self._labels_dict)
		pickle.dump(self._scale, open("models/scale.sav", "wb"))
		pickle.dump(self._kmeans, open("models/kmeans.sav", 'wb'))
		pickle.dump(self.estimator, open("models/estimator.sav", 'wb'))
		print("Done persist!")

	def save_model(self):
		"""Lưu model để nộp
		(không lưu được xfeat)
		"""

		pickle.dump(self, open("models/bovw.sav", "wb"))
		print("Saved to models/bovw.sav")

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

		kp = self.xfeat.detect(image_temp, None)
		return self.xfeat.compute(image, kp)

	def __get_files(self, path, is_balance=False):
		"""Get images dictionary from path
			Keys is label
			Values is list of images
		"""

		images = {}
		n_sample, count = 0, 0
		for each in glob(path + "*"):
			# Extract lable from path
			label = each.replace("\\", "/").split("/")[-1]
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
				if self.verbose:
					print("Sampling category", label, k, " images")
				count += k

		if self.verbose:
			print("Fetched", count, "images")

		return images, count
