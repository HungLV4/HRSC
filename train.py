"""Trainning"""
from BOVW import *
from sklearn.svm import SVC


# Classifier
# clf = SVC(kernel="rbf", C=2.8, gamma=0.0073, cache_size=10000, probability=False)
clf = SVC(kernel="rbf", C=5, gamma=0.05, cache_size=10000, probability=False)

# Features extractior
feat = cv2.xfeatures2d.SIFT_create(250, edgeThreshold=50, contrastThreshold=0.02)
# feat = cv2.xfeatures2d.SURF_create(0, extended=True)
# feat = cv2.ORB_create(patchSize=64, edgeThreshold=10, nlevels=8)

# Model
bovw = BOVW(clf,
            feat,
            n_bags=250,
            tol=0.01,
            is_resample=True,
            is_reuse=True,
            verbose=True,
            color=None)

# Train score: 82%
acc = bovw.fit_score("images/train/")
print(acc)

# Test score: 75%
# bovw.fit("images/train/")
# acc = bovw.score("images/test/")
# print(acc)

# Percision & Recall
# print(bovw.score_percision_recall("images/test/"))

# Save for is_reuse = True
# bovw.persist()

# Save model for submit
# bovw.save_model()
