import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression(fit_intercept=True, normalize=True)
reg.fit(X, y)
reg.score(X, y)

reg.coef_
reg.intercept_
reg.predict(np.array([[3, 5]]))
