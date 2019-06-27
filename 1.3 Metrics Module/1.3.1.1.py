# Import Libraries
from sklearn.metrics import mean_absolute_error 
#----------------------------------------------------

#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
#print('Mean Absolute Error Value is : ', MAEValue)