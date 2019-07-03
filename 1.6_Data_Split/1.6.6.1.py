import numpy as np
from sklearn.model_selection import LeaveOneOut
X = np.array([1, 2, 3, 4])
y = np.array([5,6,7,8])
loo = LeaveOneOut()
loo.get_n_splits(X)
print(loo)
LeaveOneOut()
for train_index, test_index in loo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('X_train \n' , X_train)
    print('X_test \n' , X_test)
    print('y_train \n' ,y_train)
    print('y_test \n' , y_test)
    print('*********************')
