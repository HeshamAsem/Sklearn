import pandas as pd
import matplotlib.pyplot as plt
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
print(k_range)

param_grid = dict(n_neighbors=k_range)
print(param_grid)

grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)
grid.fit(X, y)

pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

print(grid.cv_results_['params'])
print(grid.cv_results_['mean_test_score'])

grid_mean_scores = grid.cv_results_['mean_test_score']
print(grid_mean_scores)



plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')



print('score : ' , grid.best_score_)
print('params : ' , grid.best_params_)
print('best : ' , grid.best_estimator_)
