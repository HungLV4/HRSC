from BOVW import *
from sklearn.svm import SVC


clf = SVC(kernel="rbf", C=2.8, gamma=0.0073, cache_size=10000, probability=False)
feat = cv2.xfeatures2d.SIFT_create(250, edgeThreshold=50, contrastThreshold=0.02)
# feat = cv2.xfeatures2d.SURF_create(0, extended=True)
# feat = cv2.ORB_create(patchSize=64, edgeThreshold=10, nlevels=8)

bovw = BOVW(clf,
            feat,
            n_bags=250,
            tol=0.01,
            is_resample=True,
            is_reuse=False,
            verbose=True,
            color=None)
bovw.fit("images/train/")
acc = bovw.score("images/test/")
print(acc)

# print(bovw.score_percision_recall("images/test/"))

# bovw.persist()
