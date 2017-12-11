import cv2
import numpy as np

img = cv2.imread('images/train/100000002/100000001_0.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

orb = cv2.xfeatures2d.SURF_create(400, nOctaveLayers=4)
# orb = cv2.ORB_create(edgeThreshold=20, nlevels=1)
kp, des = orb.detectAndCompute(gray, None)

img = cv2.drawKeypoints(gray, kp, img)

print(des.shape)

cv2.imshow("ddd", img)
cv2.waitKey()
