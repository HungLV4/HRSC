"""Testing"""
import pickle as pkl
import cv2


# Load model
model = pkl.load(open("models/bovw.sav", "rb"))

# Set features extractor
model.xfeat = cv2.xfeatures2d.SIFT_create(250, edgeThreshold=50, contrastThreshold=0.02)

# Test score
acc = model.score("images/test/")
print(acc)
