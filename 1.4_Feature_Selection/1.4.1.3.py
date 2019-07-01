from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile   , chi2 
 
data = load_breast_cancer()
X = data.data
y = data.target
X.shape
sel = SelectPercentile(score_func = chi2 , percentile = 20).fit_transform(X,y)
sel.shape

