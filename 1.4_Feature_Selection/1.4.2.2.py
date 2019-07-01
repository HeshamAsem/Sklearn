from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import GenericUnivariateSelect, chi2
X, y = load_breast_cancer(return_X_y=True)
X.shape

transformer = GenericUnivariateSelect(chi2, 'k_best', param=5)
X_new = transformer.fit_transform(X, y)

X_new.shape

transformer.get_support()
