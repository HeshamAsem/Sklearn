from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2
X, y = load_digits(return_X_y=True)
X.shape

X_new = SelectPercentile(score_func =chi2, percentile=10)
X_new.fit(X, y)
selected = X_new.transform(X)
X_new.get_support()
