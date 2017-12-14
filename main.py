from BOVW import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# clf = LogisticRegression(C=0.1, solver="lbfgs", multi_class="multinomial")
clf = SVC(kernel="rbf", C=2.8, gamma=0.0073, cache_size=10000, probability=False, random_state=111)
bovw = BOVW(clf,
            tol=0.1,
            is_resample=True,
            is_reuse=False,
            edge_threshold=50,
            contrast_threshold=0.02,
            verbose=True)
bovw.fit("images/train/")
acc = bovw.score("images/test/")
print(acc)

# print(bovw.score_percision_recall("images/test/"))

bovw.persist()
