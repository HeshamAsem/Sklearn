#Import Libraries
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error
#----------------------------------------------------

#load boston data
BostonData = load_boston()

#X Data
X = BostonData.data

#y Data
y = BostonData.target
#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)
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