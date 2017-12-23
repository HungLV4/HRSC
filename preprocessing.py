import cv2
import os
import math
import numpy as np
import xml.etree.ElementTree as ET
import rotate_image as rotate


# Method to create dir if not exists
def ensure_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)


# Config xml for relatives beatween classes
tree = ET.parse("sysdata.xml")
root = tree.getroot()

# Get class
class_dict = {}
for child in root:
	tag = child.tag

	if tag == "HRSC_Classes":
		for classes in child:
			class_id = parent_id = layer = 0
			for class_elem in classes:
				if class_elem.tag == "Class_Layer":
					layer = class_elem.text
				elif class_elem.tag == "Class_ID":
					class_id = class_elem.text
				elif class_elem.tag == "HRS_Class_ID":
					parent_id = class_elem.text

			# Only handle class layer 1
			# assign layer 2 class id to layer 1
			# Nếu là layer 1 thì lấy chính nó
			if class_id not in class_dict:
				# if layer == "2":
				# 	class_dict[class_id] = class_id
				if layer == "2":
					class_dict[class_id] = parent_id
				elif layer == "1":
					class_dict[class_id] = class_id

		break

# TODO: Edit this
# if create images for testing, only cut bounding box
# if create images for trainning, do some preprocessing
src = "Test"    # "Test" or ""

# Image after preprocessing
dest = "test/"  # "test/" or "train/"

# Dir path for image preprocessing
annotation_path = "../../HRSC2016/" + src + "Annotation"
images_path = "../../HRSC2016/" + src + "Images"

path_dict = []  # save class dir had been created, so we not have to create again
# (w_mean, h_mean) = (128, 128)

# Fetch images
for file in os.listdir(annotation_path):
	# Get image name
	image_name = file.title().lower()

	# Read xml file desc for this image
	tree = ET.parse(annotation_path + "/" + image_name)
	root = tree.getroot()

	# Loop through
	for child in root:
		tag = child.tag

		# How many ship in image?
		if tag == "Annotated":
			n = int(child.text)
			if n < 1:
				break

		# Get each ship
		if tag == "HRSC_Objects":
			for objs in child:
				if objs.tag != "HRSC_Object":
					break

				x_min = x_max = y_min = y_max = 0  # box
				cx = cy = hx = hy = 0  # center point, head point
				objectID = classID = 0  # objectID is unique for each ship, classID is target
				for obj in objs:
					if obj.tag == "Object_ID":
						objectID = obj.text
					elif obj.tag == "Class_ID":
						classID = obj.text
					elif obj.tag == "box_xmin":
						x_min = int(obj.text)
					elif obj.tag == "box_xmax":
						x_max = int(obj.text)
					elif obj.tag == "box_ymin":
						y_min = int(obj.text)
					elif obj.tag == "box_ymax":
						y_max = int(obj.text)
					elif obj.tag == "mbox_cx":
						cx = float(obj.text)
					elif obj.tag == "mbox_cy":
						cy = float(obj.text)
					elif obj.tag == "header_x":
						hx = float(obj.text)
					elif obj.tag == "header_y":
						hy = float(obj.text)

				if classID not in class_dict:
					continue

				classID = class_dict[classID]
				print("%s - %s" % (image_name, objectID))

				# We rotate image that // Oy
				# cos(angle) = vec * Oy / (|vec| * |Ov|)
				vec = (hx - cx, hy - cy)
				angle = math.acos(np.dot(vec, (1, 0)) / np.linalg.norm(vec)) * 180 / math.pi
				if vec[1] < 0:
					angle *= -1
				# print(angle)

				# Rotate and crop box
				image_bmp = image_name.replace(".xml", ".bmp")
				image = cv2.imread(images_path + "/" + image_bmp)
				if dest == "train/":
					(h, w) = image.shape[:2]
					image = rotate.rotate_cus(image, angle, (cx, cy))
					(cx, cy) = rotate.rotate_point(cx, cy, cx, cy, -angle)
					(hx, hy) = rotate.rotate_point(hx, hy, cx, cy, -angle)
					x_min = math.ceil(cx - (hx - cx))
					x_max = math.ceil(hx)
					y_min = math.ceil(cy - 0.25 * (hx - cx))
					y_max = math.ceil(hy + 0.25 * (hx - cx))

					if x_min < 0:
						x_min = 0
					if y_min < 0:
						y_min = 0

				# Cut
				image = image[y_min:y_max, x_min:x_max]

				# Save image to correct class path
				image_path = "images/" + dest + "" + str(classID)
				if classID not in path_dict:
					ensure_dir(image_path)
					path_dict.append(classID)

				# Create image
				if dest == "train/":
					image_root = image

					(h, w) = image.shape[:2]
					a = w * 1.0 / h
					w_mean, h_mean = None, None
					if h > 250:
						h_mean = 250
						w_mean = math.ceil(a * h_mean)
						w = w_mean
					if w > 250:
						w_mean = 250
						h_mean = math.ceil(w_mean / a)

					if w_mean is not None and h_mean is not None:
						image = cv2.resize(image, (w_mean, h_mean), interpolation=cv2.INTER_AREA)

					# (h, w) = image.shape[:2]
					# while h <= 30 or w <= 30:
					# 	image = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_AREA)
					# 	(h, w) = image.shape[:2]

					# Rotate to generate more images
					cv2.imwrite(image_path + "/" + objectID + "_0.png", image)

					# image = image_root
					# image = rotate.rotate_about_center(image, 90)
					# (h, w) = image.shape[:2]
					#
					# y_min = math.floor(h / 6)
					# y_max = 5 * y_min
					# x_min = math.floor(w / 6)
					# x_max = 5 * x_min
					# image = image[y_min:y_max, x_min:x_max]

					# (h, w) = image.shape[:2]
					# while h <= 30 or w <= 30:
					# 	image = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_AREA)
					# 	(h, w) = image.shape[:2]

					# cv2.imwrite(image_path + "/" + objectID + "_1.png", image)
				else:
					(h, w) = image.shape[:2]
					a = w * 1.0 / h
					w_mean, h_mean = None, None
					if h > 250:
						h_mean = 250
						w_mean = math.ceil(a * h_mean)
						w = w_mean
					if w > 250:
						w_mean = 250
						h_mean = math.ceil(w_mean / a)

					if w_mean is not None and h_mean is not None:
						image = cv2.resize(image, (w_mean, h_mean), interpolation=cv2.INTER_AREA)

					cv2.imwrite(image_path + "/" + objectID + "_0.png", image)
				# if dest == "train/":
				# 	angle_arr = [0]
				# else:
				# 	angle_arr = [0]
				# for a in angle_arr:
				# 	image_m = rotate.rotate_about_center(image, a)
				# 	cv2.imwrite(image_path + "/" + objectID + "_" + str(a) + ".png", image_m)

			break
