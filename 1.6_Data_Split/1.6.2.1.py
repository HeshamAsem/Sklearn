#Import Libraries
from sklearn.model_selection import KFold
#----------------------------------------------------

#KFold Splitting data

kf = KFold(n_splits=4, random_state=44, shuffle =True)

#KFold Data
for train_index, test_index in kf.split(X):
    print('Train Data is : \n', train_index)
    print('Test Data is  : \n', test_index)
    print('-------------------------------')
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('X_train Shape is  ' , X_train.shape)
    print('X_test Shape is  ' , X_test.shape)
    print('y_train Shape is  ' ,y_train.shape)
    print('y_test Shape is  ' , y_test.shape)
    print('========================================')