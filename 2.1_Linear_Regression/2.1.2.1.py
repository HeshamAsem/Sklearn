#Import Libraries
from sklearn.linear_model import Ridge
#----------------------------------------------------

#Applying Ridge Regression Model 

'''
#sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False,
#                           copy_X=True, max_iter=None, tol=0.001, solver='auto',
#                           random_state=None)
'''

RidgeRegressionModel = Ridge(alpha=1.0,random_state=33)
RidgeRegressionModel.fit(X_train, y_train)

#Calculating Details
print('Ridge Regression Train Score is : ' , RidgeRegressionModel.score(X_train, y_train))
print('Ridge Regression Test Score is : ' , RidgeRegressionModel.score(X_test, y_test))
print('Ridge Regression Coef is : ' , RidgeRegressionModel.coef_)
print('Ridge Regression intercept is : ' , RidgeRegressionModel.intercept_)
print('----------------------------------------------------')

#Calculating Prediction
y_pred = RidgeRegressionModel.predict(X_test)
print('Predicted Value for Ridge Regression is : ' , y_pred[:10])