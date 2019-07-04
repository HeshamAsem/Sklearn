#Import Libraries
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error
#----------------------------------------------------

#Applying Lasso Regression Model 

'''
#sklearn.linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=
#                           False, copy_X=True, max_iter=1000, tol=0.0001,
#                           warm_start=False, positive=False, random_state=None,selection='cyclic')
'''

LassoRegressionModel = Lasso(alpha=1.0,random_state=33,normalize=False)
LassoRegressionModel.fit(X_train, y_train)

#Calculating Details
print('Lasso Regression Train Score is : ' , LassoRegressionModel.score(X_train, y_train))
print('Lasso Regression Test Score is : ' , LassoRegressionModel.score(X_test, y_test))
print('Lasso Regression Coef is : ' , LassoRegressionModel.coef_)
print('Lasso Regression intercept is : ' , LassoRegressionModel.intercept_)
print('----------------------------------------------------')

#Calculating Prediction
y_pred = LassoRegressionModel.predict(X_test)
print('Predicted Value for Lasso Regression is : ' , y_pred[:10])

#----------------------------------------------------
#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue)

#----------------------------------------------------
#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)

#----------------------------------------------------
#Calculating Median Squared Error
MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue )