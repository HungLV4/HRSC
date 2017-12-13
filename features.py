import cv2
import numpy as np

from skimage.feature import hog


img = cv2.imread('images/train/100000002/100000001_0.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# gray = img
print(gray.shape)
orb = cv2.xfeatures2d.SIFT_create(200, edgeThreshold=50, contrastThreshold=0.02)
# orb = cv2.xfeatures2d.SURF_create(extended=True)
# orb = cv2.ORB_create(patchSize=64, edgeThreshold=10, nlevels=8)
kp = orb.detect(gray, None)
kp, des = orb.compute(img, kp)
# kp, des = orb.detectAndCompute(gray, None)
print(des.shape)
img = cv2.drawKeypoints(img, kp, img)

# feat, img = hog(gray, visualise=True, transform_sqrt=True)

# print(feat.shape)

cv2.imshow("ddd", img)
cv2.waitKey()
