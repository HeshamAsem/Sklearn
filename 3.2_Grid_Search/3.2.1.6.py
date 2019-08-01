import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


iris = load_iris()


X = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)

k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']


param_grid = dict(n_neighbors=k_range, weights=weight_options)
print(param_grid)


grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)
grid.fit(X, y)

pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

print(grid.best_score_)
print(grid.best_params_)


knn = KNeighborsClassifier(n_neighbors=13, weights='uniform')
knn.fit(X, y)


knn.predict([[3, 5, 4, 2]])
 

