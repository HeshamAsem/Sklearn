#Import Libraries
from sklearn.metrics import median_absolute_error
#----------------------------------------------------


#Calculating Median Squared Error
MdSEValue = median_absolute_error(y_test, y_pred)
#print('Median Squared Error Value is : ', MdSEValue )