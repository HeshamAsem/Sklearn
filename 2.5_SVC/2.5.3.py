import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
#y = np.array(['a','a','b','b'])
y = np.array([1,1,2,2])
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(X, y)
print(clf.predict([[-0.8, -1]]))