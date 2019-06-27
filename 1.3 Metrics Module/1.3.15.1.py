#Import Libraries
from sklearn.metrics import zero_one_loss
#----------------------------------------------------

#Calculating Zero One Loss:  
#zero_one_loss(y_true, y_pred, normalize = True, sample_weight = None)

ZeroOneLossValue = zero_one_loss(y_test,y_pred,normalize=False) 
#print('Zero One Loss Value : ', ZeroOneLossValue )