import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0,0,1,1])
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,random_state=36851234)
for train_index, test_index in rskf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('X_train \n' , X_train)
    print('X_test \n' , X_test)
    print('y_train \n' ,y_train)
    print('y_test \n' , y_test)
    print('*********************')
