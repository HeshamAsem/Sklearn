#Import Libraries
from sklearn.metrics import accuracy_score
#----------------------------------------------------


#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))
AccScore = accuracy_score(y_test, y_pred, normalize=False)
#print('Accuracy Score is : ', AccScore)