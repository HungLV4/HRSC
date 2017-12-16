from BOVW import *
from sklearn.svm import SVC


clf = SVC(kernel="rbf", C=2.8, gamma=0.0073, cache_size=10000, probability=False, random_state=111)
feat = cv2.xfeatures2d.SIFT_create(300, edgeThreshold=50, contrastThreshold=0.02)

bovw = BOVW(clf,
            feat,
            n_bags=300,
            tol=0.1,
            is_resample=True,
            is_reuse=False,
            verbose=True)
bovw.fit("images/train/")
acc = bovw.score("images/test/")
print(acc)

# print(bovw.score_percision_recall("images/test/"))

# bovw.persist()
