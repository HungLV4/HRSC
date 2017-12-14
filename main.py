from BOVW import *
from sklearn.svm import SVC


clf = SVC(kernel="rbf", C=2.8, gamma=0.0073, cache_size=10000, probability=True, random_state=111)
bovw = BOVW(clf, tol=0.1, is_resample=True, is_reuse=True, verbose=True)
bovw.fit("images/train/")
acc = bovw.score("images/test/")
print(acc)

# print(bovw.score_percision_recall("images/test/"))
