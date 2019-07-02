
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)


poly = PolynomialFeatures(degree=2 , include_bias = True)
poly.fit_transform(X)



poly = PolynomialFeatures(interaction_only=True)
poly.fit_transform(X)
