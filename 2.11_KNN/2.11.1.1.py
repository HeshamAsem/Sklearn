#Import Libraries
from sklearn.neighbors import KNeighborsRegressor
#----------------------------------------------------

#Applying KNeighborsRegressor Model 

'''
#sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights=, algorithm=’auto’, leaf_size=30,
#                                      p=2, metric=’minkowski’, metric_params=None,n_jobs=None)
'''

KNeighborsRegressorModel = KNeighborsRegressor(n_neighbors = 5, weights='uniform', #also can be : distance, or defined function 
                                               algorithm = 'auto')    #also can be : ball_tree ,  kd_tree  , brute
KNeighborsRegressorModel.fit(X_train, y_train)

#Calculating Details
print('KNeighborsRegressorModel Train Score is : ' , KNeighborsRegressorModel.score(X_train, y_train))
print('KNeighborsRegressorModel Test Score is : ' , KNeighborsRegressorModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = KNeighborsRegressorModel.predict(X_test)
print('Predicted Value for KNeighborsRegressorModel is : ' , y_pred[:10])