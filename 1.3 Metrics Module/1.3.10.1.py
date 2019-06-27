#Import Libraries
from sklearn.metrics import precision_recall_curve
#----------------------------------------------------

#Calculating Precision recall Curve :  
# precision_recall_curve(y_true, probas_pred, pos_label=None, sample_weight=None)

PrecisionValue, RecallValue, ThresholdsValue = precision_recall_curve(y_test,y_pred)
#print('Precision Value is : ', PrecisionValue)
#print('Recall Value is : ', RecallValue)
#print('Thresholds Value is : ', ThresholdsValue)