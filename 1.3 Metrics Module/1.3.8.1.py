#Import Libraries
from sklearn.metrics import precision_score
#----------------------------------------------------

#Calculating Precision Score : (Specificity) #(TP / float(TP + FP))  
# precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’,sample_weight=None)

PrecisionScore = precision_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples
#print('Precision Score is : ', PrecisionScore)