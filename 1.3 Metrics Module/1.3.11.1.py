#Import Libraries
from sklearn.metrics import classification_report
#----------------------------------------------------

#Calculating classification Report :  
#classification_report(y_true, y_pred, labels=None, target_names=None,sample_weight=None, digits=2, output_dict=False)

ClassificationReport = classification_report(y_test,y_pred)
#print('Classification Report is : ', ClassificationReport )