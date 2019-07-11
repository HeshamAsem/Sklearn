from sklearn.svm import SVR
import numpy as np
n_samples=10
n_features = 10
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
X.shape
y
clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
clf.fit(X, y)

newx = np.random.randn(1,10)
y_pred = clf.predict(newx)

print(newx , ' \n ' ,y_pred)