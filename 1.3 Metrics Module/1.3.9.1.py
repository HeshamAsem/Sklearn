#Import Libraries
from sklearn.metrics import precision_recall_fscore_support
#----------------------------------------------------

#Calculating Precision recall Score :  
#metrics.precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=None, pos_label=1, average=
#                                        None, warn_for = ('precision’,’recall’, ’f-score’), sample_weight=None)

PrecisionRecallScore = precision_recall_fscore_support(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples
#print('Precision Recall Score is : ', PrecisionRecallScore)