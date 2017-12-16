import pickle as pkl
import cv2


model = pkl.load(open("models/bovw.sav", "rb"))
model.xfeat = cv2.xfeatures2d.SIFT_create(250, edgeThreshold=50, contrastThreshold=0.02)
acc = model.score("images/test/")
print(acc)
