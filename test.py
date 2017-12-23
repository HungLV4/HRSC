"""Testing"""
import pickle as pkl
import cv2


# Load model
model = pkl.load(open("models/HRSC_new.sav", "rb"))

# Set features extractor
model.xfeat = cv2.xfeatures2d.SIFT_create(250, edgeThreshold=50, contrastThreshold=0.02)
model.is_resample = False

# Test score
# acc = model.score("images/test/")
# print(acc)

# Percision & Recall
model.confusion_matrix("images/test/")
