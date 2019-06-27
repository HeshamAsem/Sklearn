#Import Libraries
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
#----------------------------------------------------

#Calculating Area Under the Curve :  

fprValue2, tprValue2, thresholdsValue2 = roc_curve(y_test,y_pred)
AUCValue = auc(fprValue2, tprValue2)
#print('AUC Value  : ', AUCValue)