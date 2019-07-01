from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
data = load_breast_cancer()
X = data.data
y = data.target


sel = SelectFromModel(RandomForestClassifier(n_estimators = 20)) 
sel.fit(X,y)
selected_features = sel.transform(X)
sel.get_support()
