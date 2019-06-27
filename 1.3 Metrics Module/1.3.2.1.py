#Import Libraries
from sklearn.metrics import mean_squared_error 
#----------------------------------------------------


#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
#print('Mean Squared Error Value is : ', MSEValue)