"""Trainning"""
from BOVW import *
from sklearn.svm import SVC


# Classifier
clf = SVC(kernel="rbf", C=5, gamma=0.05, cache_size=10000, probability=False, class_weight="balanced")
# clf = SVC(kernel="rbf", C=5, gamma=0.05, cache_size=10000, probability=False)

# Features extractior
feat = cv2.xfeatures2d.SIFT_create(250, edgeThreshold=50, contrastThreshold=0.02)

# Model
bovw = BOVW(clf,
            feat,
            n_bags=250,
            is_resample=False,
            is_reuse=False,
            mini_batches=True,
            verbose=True)

# Test score
# bovw.fit("images/train/")
# acc = bovw.score("images/test/")
# print(acc)

# Percision & Recall
# bovw.confusion_matrix("images/test/")

# Cross Validation
acc = bovw.cross_validation(["images/train/", "images/test/"])
print(acc)

# Save for is_reuse = True
# bovw.persist()

# Save model for submit
# bovw.save_model("HRSC")
