#Import Libraries
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error
#----------------------------------------------------

#Applying SGDRegressor Model 

'''
#sklearn.linear_model.SGDRegressor(loss='squared_loss’, penalty=’l2’, alpha=0.0001,
#                                  l1_ratio=0.15, fit_intercept=True, max_iter=None,
#                                  tol=None, shuffle=True, verbose=0, epsilon=0.1,
#                                  random_state=None, learning_rate='invscaling’,
#                                  eta0=0.01, power_t=0.25, early_stopping=False,
#                                  validation_fraction=0.1, n_iter_no_change=5,
#                                  warm_start=False, average=False, n_iter=None)
'''

SGDRegressionModel = SGDRegressor(alpha=0.1,random_state=33,penalty='l2',loss = 'huber')
SGDRegressionModel.fit(X_train, y_train)

#Calculating Details
print('SGD Regression Train Score is : ' , SGDRegressionModel.score(X_train, y_train))
print('SGD Regression Test Score is : ' , SGDRegressionModel.score(X_test, y_test))
print('SGD Regression Coef is : ' , SGDRegressionModel.coef_)
print('SGD Regression intercept is : ' , SGDRegressionModel.intercept_)
print('----------------------------------------------------')

#Calculating Prediction
y_pred = SGDRegressionModel.predict(X_test)
print('Predicted Value for SGD Regression is : ' , y_pred[:10])

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