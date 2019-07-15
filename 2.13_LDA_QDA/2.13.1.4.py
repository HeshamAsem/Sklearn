import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.random.randint(20, size=(50, 10))
y = np.random.randint(5, size=(50, 1))

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

clf.score(X,y)
z = np.random.randint(20, size=(1, 10))

print(clf.predict(z))


